from __future__ import division, print_function
# from torchsummary import summary
import torch
from torch import cuda
import  torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torch.autograd import Variable
import random
from mnist import loadMnistRot, random_rotation, linear_interpolation_2D
from torchvision import transforms

import sys
sys.path.append('../') #Import
from layers_2D import *
from utils import getGrid

#!/usr/bin/env python
__author__ = "Anders U. Waldeland"
__email__ = "anders@nr.no"

"""
A reproduction of the MNIST-classification network described in:
Rotation equivariant vector field networks (ICCV 2017)
Diego Marcos, Michele Volpi, Nikos Komodakis, Devis Tuia
https://arxiv.org/abs/1612.09346
https://github.com/dmarcosg/RotEqNet
"""

img_size = (540, 960)

if __name__ == '__main__':

    # Define network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.main = nn.Sequential(
                nn.Conv2d(1, 8, [3, 3], 1, 3 // 2),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(2),

                nn.Conv2d(8, 12, [3, 3], 1, 3 // 2),
                nn.ReLU(),
                nn.BatchNorm2d(12),
                nn.MaxPool2d(2),

                nn.Conv2d(12, 8, [3, 3], 1, 3 // 2),
                nn.ReLU(),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(2),

                nn.ConvTranspose2d(8, 4, [3, 3], 2, (0,1)),
                nn.ReLU(),
                nn.BatchNorm2d(4),

                nn.ConvTranspose2d(4, 2, [3, 3], 2),
                nn.ReLU(),
                nn.BatchNorm2d(2),
                nn.ConvTranspose2d(2, 1, [3, 3], 2),
                nn.UpsamplingBilinear2d(size=(540, 960))
            )

        def forward(self, x):
            x = self.main(x)
            x = F.sigmoid(x)
            # x = x.view(x.size()[0], x.size()[1])

            return x


    gpu_no =  0 # Set to False for cpu-version

    #Setup net, loss function, optimizer and hyper parameters

    net = Net()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    if type(gpu_no) == int:
        net.cuda(gpu_no)

    if True: #Current best setup using this implementation - error rate of 1.2%
        start_lr = 0.01
        batch_size = 2
        optimizer = optim.Adam(net.parameters(), lr=start_lr)  # , weight_decay=0.01)
        use_test_time_augmentation = True
        use_train_time_augmentation = True

    def rotate_im(im, theta):
        grid = getGrid([28, 28])
        grid = rotate_grid_2D(grid, theta)
        grid += 13.5
        data = linear_interpolation_2D(im, grid)
        data = np.reshape(data, [28, 28])
        return data.astype('float32')


    def test(model, dataset, mode):
        """ Return test-acuracy for a dataset"""
        model.eval()

        true = []
        pred = []
        for batch_no in range(len(dataset) // batch_size):
            data, labels = getBatch(dataset, mode)

            #Run same sample with different orientations through network and average output
            if use_test_time_augmentation and mode == 'test':
                data = data.cpu()
                original_data = data.clone().data.cpu().numpy()

                out = None
                rotations = [0,15,30,45, 60, 75, 90]

                for rotation in rotations:

                    for i in range(batch_size):
                        im = original_data[i,:,:,:].squeeze()
                        #im = rotate_im(im, rotation)
                        im = im.reshape([1, 1, img_size])
                        im = torch.FloatTensor(im)
                        data[i,:,:,:] = im
                        #data = im

                    if type(gpu_no) == int:
                        data = data.cuda(gpu_no)

                    if out is None:
                        out = F.softmax(model(data),dim=1)
                    else:
                        out += F.softmax(model(data),dim=1)

                out /= len(rotations)

            #Only run once
            else:
                out = F.softmax(model(data),dim=1)

            loss = criterion(out, labels)
            _, c = torch.max(out, 1)
            true.append(labels.data.cpu().numpy()[:, 0, :, :])
            pred.append(c.data.cpu().numpy())
        true = np.concatenate(true, 0)
        pred = np.concatenate(pred, 0)
        true = np.squeeze(true)
        print(true.shape, pred.shape)
        # pred = np.round(pred)
        acc = np.average(np.isclose(pred, true, atol=0.5))
        return acc

    def getBatch(dataset, mode):
        """ Collect a batch of samples from list """

        # Make batch
        data = []
        labels = []
        for sample_no in range(batch_size):
            tmp = dataset.pop()  # Get top element and remove from list
            img = tmp[0].astype('float32').squeeze()

            # Train-time random rotation
            #if mode == 'train' and use_train_time_augmentation:
            #    img = random_rotation(img)

            data.append(np.expand_dims(np.expand_dims(img, 0), 0))
            labels.append(tmp[1].squeeze())
        data = np.concatenate(data, 0)
        labels = np.array(labels, 'float32')

        data = Variable(torch.from_numpy(data))
        labels = Variable(torch.from_numpy(labels))

        if type(gpu_no) == int:
            data = data.cuda(gpu_no)
            labels = labels.cuda(gpu_no)
        return data, labels

    def adjust_learning_rate(optimizer, epoch):
        """Gradually decay learning rate"""
        if epoch == 20:
            lr = start_lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 40:
            lr = start_lr / 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 60:
            lr = start_lr / 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    #Load datasets
    train_set, val_set,  test_set = loadMnistRot()
    best_acc = 0
    for epoch_no in range(10):

        #Random order for each epoch
        train_set_for_epoch = train_set[:] #Make a copy
        random.shuffle(train_set_for_epoch) #Shuffle the copy

        #Training
        net.train()
        for batch_no in range(len(train_set)//batch_size):

            # Train
            optimizer.zero_grad()

            data, labels = getBatch(train_set_for_epoch, 'train')
            out = net( data )
            loss = criterion( out,labels )
            _, c = torch.max(out, 1)
            loss.backward()

            optimizer.step()

            #Print training-acc
            if batch_no%10 == 0:
                print('Train', 'epoch:', epoch_no,
                      ' batch:', batch_no,
                      ' loss:', loss.data.cpu().numpy(),
                      #' acc:', np.average((c == labels).data.cpu().numpy())
                      )


        #Validation
        # acc = test(net, val_set[:], 'val')
        # print('Val',  'epoch:', epoch_no,  ' acc:', acc)
        #
        # #Save model if better than previous
        # if acc > best_acc:
        #     torch.save(net.state_dict(), 'best_model.pt')
        #     best_acc = acc
        #     print('Model saved')

        adjust_learning_rate(optimizer, epoch_no)

    # Finally test on test-set with the best model
    # net.load_state_dict(torch.load('best_model.pt'))
    # net.cuda()
    # print('Test', 'acc:', test(net, test_set[:], 'test'))
    net.eval()
    from PIL import Image

    loader = transforms.Compose([transforms.ToTensor()])
    image = Image.fromarray(train_set[50][0][0, :, :])
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    image = image.cuda()
    xyz = net(image)
    xyz = xyz.data.cpu().numpy()
    xyz = np.squeeze(xyz)
    xyz = Image.fromarray(xyz[:, :]*255)
    mask = Image.fromarray(train_set[50][1][0, :, :]*255)
    orig = Image.fromarray(train_set[50][0][0, :, :]*255)
    print(xyz.show(title='net'))
    print(mask.show(title='mask'))
    print(orig.show(title='orig'))
