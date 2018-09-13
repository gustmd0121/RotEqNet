# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torch import optim
import numpy as np
from torch.autograd import Variable
import random
from torchvision import transforms
import os
import sys
from random import randint
# Local imports
# Layers
from framework.layers import RotConv
from framework.layers import VectorUpsample
from framework.layers import ReturnU
from framework.layers import VectorBatchNorm
from framework.layers import SpatialPooling
from framework.layers import OrientationPooling
from framework.loss import F1Loss
# Utils
from framework.utils.utils import *

"""
An implementation to detect Dung Beetles and their orientation based on the concept proposed in:
Rotation equivariant vector field networks (ICCV 2017)
Diego Marcos, Michele Volpi, Nikos Komodakis, Devis Tuia
https://arxiv.org/abs/1612.09346
https://github.com/dmarcosg/RotEqNet
"""

if __name__ == '__main__':
    # Define network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            filter_size = 9

            self.main = nn.Sequential(

                RotConv(1, 8, [filter_size, filter_size], 1, filter_size // 2, n_angles=17, mode=1),
                OrientationPooling(),
                SpatialPooling(2),

                RotConv(8, 12, [filter_size, filter_size], 1, filter_size // 2, n_angles=17, mode=2),
                OrientationPooling(),
                SpatialPooling(2),

                RotConv(12, 8, [filter_size, filter_size], 1, filter_size // 2, n_angles=17, mode=2),
                OrientationPooling(),
                VectorBatchNorm(8),
                VectorUpsample(scale_factor=2),

                RotConv(8, 4, [filter_size, filter_size], 1, filter_size // 2, n_angles=17, mode=2),
                OrientationPooling(),
                VectorBatchNorm(4),
                VectorUpsample(scale_factor=2),

                RotConv(4, 2, [filter_size, filter_size], 1, filter_size // 2, n_angles=17, mode=2),
                OrientationPooling(),
                VectorBatchNorm(2),

                RotConv(2, 1, [filter_size, filter_size], 1, filter_size // 2, n_angles=17, mode=2),
                OrientationPooling(),
                VectorUpsample(size=img_size),

                ReturnU()
            )

        def forward(self, x):
            x = self.main(x)
            x = F.sigmoid(x)
            return x


    def adjust_learning_rate(optimizer, epoch):
        """
        Gradually decay learning rate"
        NOTE: Will be used in future implementation
        :param optimizer: optimizer which should be used
        :param epoch: current number of training epoch
        """""
        if epoch == 4:
            lr = start_lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 6:
            lr = start_lr / 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 8:
            lr = start_lr / 100
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


    def getBatch(dataset):
        """ Collect a batch of samples from list """

        # Make batch
        data = []
        labels = []
        for sample_no in range(batch_size):
            tmp = dataset.pop()  # Get top element and remove from list
            img = tmp[0].astype('float32').squeeze()

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


    def load_data(train, test):
        imgs = np.load(base_folder + train + "/" + train + "_input.npz")['data']
        imgs = np.split(imgs, imgs.shape[0], 0)

        for i in range(len(imgs)):
            imgs[i] = imgs[i] / 255 - 0.5

        mask_data = np.load(base_folder + test + "/" + test + "_masks.npz")
        mask_data = np.split(mask_data['beetle'], mask_data['beetle'].shape[2], 2)

        first_rand = 30 #randint(0, len(imgs)-1)
        second_rand = 42 #randint(0, len(imgs)-1)

        test_data = list(zip([imgs[first_rand], imgs[second_rand]], [mask_data[first_rand], mask_data[second_rand]]))
        del imgs[first_rand]
        del imgs[second_rand]

        train_data = list(zip(imgs, mask_data))

        return train_data, train_data, test_data


    def train(net):
        # Net Parameters
        if type(gpu_no) == int:
            net.cuda(gpu_no)

        optimizer = optim.Adam(net.parameters(), lr=start_lr)  # , weight_decay=0.01)

        for epoch_no in range(epoch_size):

            # Random order for each epoch
            train_set_for_epoch = train_set[:]  # Make a copy
            random.shuffle(train_set_for_epoch)  # Shuffle the copy

            # Training
            net.train()
            for batch_no in range(len(train_set) // batch_size):

                # Train
                optimizer.zero_grad()

                data, labels = getBatch(train_set_for_epoch)
                out = net(data)
                loss = criterion(out, labels)
                loss.backward()

                optimizer.step()

                # Print training-acc
                if batch_no % 10 == 0:
                    print('Train', 'epoch:', epoch_no,
                          ' batch:', batch_no,
                          ' loss:', loss.data.cpu().numpy()
                          )

            adjust_learning_rate(optimizer, epoch_no)
            torch.save(net.state_dict(), model_file)


    def load(net):
        net.load_state_dict(torch.load(model_file))
        if type(gpu_no) == int:
            net.cuda()


    def test(net):
        loader = transforms.Compose([transforms.ToTensor()])
        image = Image.fromarray(test_set[test_image][0][0, :, :])
        image = loader(image).float()
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)
        image = image.cuda()
        xyz = net(image)
        xyz = xyz.data.cpu().numpy()
        xyz = np.squeeze(xyz)
        xyz = Image.fromarray(xyz[:, :] * 255)
        orig = Image.fromarray((test_set[test_image][0][0, :, :] + 0.5) * 255)
        mask = Image.fromarray(test_set[test_image][1][:, :, 0] * 255)
        xyz.show(title='net')
        mask.show(title='mask')
        orig.show(title='orig')


    # ------MAIN------
    # Load datasets
    img_size = (300, 400)
    base_folder = "./data/"
    # workaround
    if not os.path.isdir(base_folder):
        base_folder = "." + base_folder
    train_file = "Allogymnopleuri_#05"
    test_file = "Allogymnopleuri_#05"
    train_set, val_set, test_set = load_data(train_file, test_file)
    model_file = train_file + "_model.pt"
    if (len(sys.argv) == 4):
        model_file = train_file + "_" + sys.argv[3] + ".pt"

    # Setup net, loss function, optimizer and hyper parameters
    start_lr = 0.01
    epoch_size = 2
    if (len(sys.argv) > 2 and sys.argv[1] == "train"):
        epoch_size = (int)(sys.argv[2])
    batch_size = 5
    test_image = 0
    if (len(sys.argv) > 2 and sys.argv[1] == "test"):
        test_image = (int)(sys.argv[2])

    criterion = nn.BCELoss()
    net = Net()
    gpu_no = 0  # Set to False for cpu-version

    if (len(sys.argv) == 1):
        train(net)
        test(net)
    elif (sys.argv[1] == "train"):
        train(net)
    elif (sys.argv[1] == "test"):
        load(net)
        test(net)
