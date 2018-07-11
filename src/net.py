# Global imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
from torch.autograd import Variable
import random
from torchvision import transforms
import os
import math
from math import hypot
import matplotlib.pyplot as plt
# Local imports
# Layers
from framework import RotConv
from framework.layers import VectorUpsample
from framework.layers import VectorToMagnitude
from framework.layers import VectorBatchNorm
from framework.layers import SpatialPooling
from framework.layers import OrientationPooling
# Utils
from framework.utils.utils import *

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

epoch_size = 2
batch_size = 3
num_image = 50
train_file = "Allogymnopleuri_#05"
test_file = "Allogymnopleuri_#05"
base_folder = "./data/"
if not os.path.isdir(base_folder):
    base_folder = "." + base_folder
img_size = (300, 400)

if __name__ == '__main__':

    # Define network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.main = nn.Sequential(
                RotConv(1, 4, [3, 3], 1, 3 // 2, n_angles=17, mode=1),
                OrientationPooling(),
                VectorBatchNorm(4),
                SpatialPooling(2),

                RotConv(4, 8, [3, 3], 1, 3 // 2, n_angles=17, mode=2),
                OrientationPooling(),
                VectorBatchNorm(8),
                SpatialPooling(2),

                # RotConv(12, 8, [3, 3], 1, 3 // 2, n_angles=17, mode=2),
                # OrientationPooling(),
                # VectorBatchNorm(8),
                # VectorUpsample(scale_factor=2),

                RotConv(8, 4, [3, 3], 1, 3 // 2, n_angles=17, mode=2),
                OrientationPooling(),
                VectorBatchNorm(4),
                VectorUpsample(scale_factor=2),

                RotConv(4, 2, [3, 3], 1, 3 // 2, n_angles=17, mode=2),
                OrientationPooling(),
                VectorBatchNorm(2),
                VectorUpsample(size=img_size),

                RotConv(2, 1, [3, 3], 1, 3 // 2, n_angles=17, mode=2),
                OrientationPooling(),


                VectorToMagnitude()
            )

        def forward(self, x):
            x = self.main(x)
            y = F.sigmoid(x[0])
            z = F.relu(x[1])
            return (y, z)


    gpu_no =  0 # Set to False for cpu-version

    #Setup net, loss function, optimizer and hyper parameters
    net = Net()

    # Net Parameters
    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()
    if type(gpu_no) == int:
        net.cuda(gpu_no)

    start_lr = 0.01
    optimizer = optim.Adam(net.parameters(), lr=start_lr)  # , weight_decay=0.01)
    use_test_time_augmentation = True
    use_train_time_augmentation = True


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

            data.append(np.expand_dims(img, 0))
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


    def load_data(train, test):
        # trainfiles
        imgs =  np.load(base_folder + train + "/" + train + "_input.npz")['data']
        # print(imgs.shape)
        #imgs = np.split(imgs, imgs.shape[0],0)
        for i in range(len(imgs)):
            imgs[i] = imgs[i] / 255 -0.5

        np.swapaxes(imgs, 0, 1)
        mask_data = np.load(base_folder + train + "/" + train + "_masks.npz")['beetle']
        for i in range(len(mask_data)):
            mask_data[i][1] = mask_data[i][1] * math.pi / 180
        # print(mask_data.shape)
        #mask_data = np.split(mask_data['beetle'], mask_data['beetle'].shape[0],0)
        # for i in range(len(mask_data)):
        #     mask_data[i] = np.squeeze(np.stack([mask_data[i], mask_data[i] * -1 + np.ones(mask_data[i].shape)], axis=0))
        train = list(zip(imgs, mask_data))


        # testfiles
        imgs =  np.load(base_folder + test + "/" + test + "_input.npz")['data']
        #imgs = np.split(imgs, imgs.shape[0],0)
        for i in range(len(imgs)):
            imgs[i] = imgs[i] / 255 -0.5

        mask_data = np.load(base_folder + test + "/" + test + "_masks.npz")['beetle']
        for i in range(len(mask_data)):
            mask_data[i][1] = mask_data[i][1] * math.pi / 180
        #mask_data = np.split(mask_data['beetle'], mask_data['beetle'].shape[0],0)
        # for i in range(len(mask_data)):
        #     mask_data[i] = np.squeeze(np.stack([mask_data[i], mask_data[i] * -1 + np.ones(mask_data[i].shape)], axis=0))
        test = list(zip(imgs, mask_data))

        return train, train, test

    #Load datasets
    train_set, val_set,  test_set = load_data(train_file, test_file)
    best_acc = 0
    for epoch_no in range(epoch_size):

        #Random order for each epoch
        train_set_for_epoch = train_set[:] #Make a copy
        random.shuffle(train_set_for_epoch) #Shuffle the copy

        #Training
        net.train()
        for batch_no in range(len(train_set)//batch_size):

            # Train
            optimizer.zero_grad()

            data, labels = getBatch(train_set_for_epoch, 'train')
            out1, out2 = net( data )
            loss1 = criterion1( out1.squeeze(1),labels[:, 0, :, :] )
            loss2 = criterion2( out2.squeeze(1),labels[:, 1, :, :] )
            loss = loss1 + loss2
            # print(loss1, loss2/ 360)
            #_, c = torch.max(out1, 1)
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
    # image = Image.fromarray(test_set[50][0])
    # list_shape(test_set[50][0])
    print(test_set[num_image][0][0].shape)
    image = loader(np.expand_dims(test_set[num_image][0][0], 2)).float()
    # image.squeeze_()
    # image.unsqueeze_(0)
    #u = image[:,0,:,:]
    #u.unsqueeze_(0)
    #print(u.shape)
    # image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    image = image.cuda()
    xyz = net(image)
    magnitude = xyz[0].data.cpu().numpy().squeeze(0).squeeze(0)
    angles = xyz[1].data.cpu().numpy().squeeze(0).squeeze(0)
    print("Angle:",str(get_average_angle(magnitude, angles, 0.15)*180/math.pi))
    print("Real Angle:",np.max(test_set[num_image][1][1])*180/math.pi)
    print("")
    shape(angles)
    print("Angle, x=138:", test_set[num_image][1][1][138]*180/math.pi)
    print("Real Angle, x=138:", angles[138]*180/math.pi)
    magnitude = np.squeeze(magnitude)
    mag = Image.fromarray((magnitude*255))
    print(mag.show(title='net'))
    tresh = Image.fromarray(treshhold(magnitude, 0.15)*255)
    print(tresh.show(title='tresh'))
    orig = Image.fromarray((test_set[num_image][0][0]+0.5)*255)
    print(orig.show(title='orig'))

import numpy as np
import matplotlib.pyplot as plt

# Set limits and number of points in grid
y, x = np.mgrid[0:len(magnitude[0]):100, 0:len(magnitude[0]):100]

x_obstacle, y_obstacle = 0.0, 0.0
alpha_obstacle, a_obstacle, b_obstacle = 1.0, 1e3, 2e3

p = -alpha_obstacle * np.exp(-((x - x_obstacle)**2 / a_obstacle
                               + (y - y_obstacle)**2 / b_obstacle))

# For the absolute values of "dx" and "dy" to mean anything, we'll need to
# specify the "cellsize" of our grid.  For purely visual purposes, though,
# we could get away with just "dy, dx = np.gradient(p)".

skip = (slice(None, None, 5), slice(None, None, 5))

fig, ax = plt.subplots()
im = ax.imshow(magnitude)

def pol2cart(magnitude_map, angle_map):
    u = magnitude_map * np.cos(angle_map)
    v = magnitude_map * np.sin(angle_map)
    return u, v

def xy_coords():
    x_single_col = np.array(list(range(0, img_size[1])))
    y_single_row = np.array(list(range(0, img_size[0])))

    x = np.tile(x_single_col, (img_size[0], 1))
    y = np.tile(y_single_row, (img_size[1], 1))
    y = y.transpose()

    return x, y

print(xy_coords)

#ax.quiver(x[skip], y[skip], angles= angles, color = 'w')

(u, v) = pol2cart(magnitude, angles)
(x, y) = xy_coords()


u = u*100
v = v*100

print(np.max(angles))

ax.quiver(x, y, u, v, color='w')
fig.colorbar(im)
ax.set(aspect=1, title='Quiver Plot')
plt.show()

# not used ("mnist.py" file)
# def linear_interpolation_2D(input_array, indices, outside_val=0, boundary_correction=True):
#     # http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy
#     output = np.empty(indices[0].shape)
#     ind_0 = indices[0,:]
#     ind_1 = indices[1,:]
#
#     N0, N1 = input_array.shape
#
#     x0_0 = ind_0.astype(np.integer)
#     x1_0 = ind_1.astype(np.integer)
#     x0_1 = x0_0 + 1
#     x1_1 = x1_0 + 1
#
#     # Check if inds are beyond array boundary:
#     if boundary_correction:
#         # put all samples outside datacube to 0
#         inds_out_of_range = (x0_0 < 0) | (x0_1 < 0) | (x1_0 < 0) | (x1_1 < 0) |  \
#                             (x0_0 >= N0) | (x0_1 >= N0) | (x1_0 >= N1) | (x1_1 >= N1)
#
#         x0_0[inds_out_of_range] = 0
#         x1_0[inds_out_of_range] = 0
#         x0_1[inds_out_of_range] = 0
#         x1_1[inds_out_of_range] = 0
#
#     w0 = ind_0 - x0_0
#     w1 = ind_1 - x1_0
#     # Replace by this...
#     # input_array.take(np.array([x0_0, x1_0, x2_0]))
#     output = (input_array[x0_0, x1_0] * (1 - w0) * (1 - w1)  +
#               input_array[x0_1, x1_0] * w0 * (1 - w1)  +
#               input_array[x0_0, x1_1] * (1 - w0) * w1  +
#               input_array[x0_1, x1_1] * w0 * w1 )
#
#
#     if boundary_correction:
#         output[inds_out_of_range] = 0
#
#     return output
#
#
# def random_rotation(data):
#     rot = np.random.rand() * 360  # Random rotation
#     grid = getGrid([28, 28])
#     grid = rotate_grid_2D(grid, rot)
#     grid += 13.5
#     data = linear_interpolation_2D(data, grid)
#     data = np.reshape(data, [28, 28])
#     data = data / float(np.max(data))
#     return data.astype('float32')

# not used ("make_mnist_rot.py" file)
# def makeMnistRot():
#     """
#     Make MNIST-rot from MNIST
#     Select all training and test samples from MNIST and select 10000 for train,
#     2000 for val and 50000 for test. Apply a random rotation to each image.
#
#     Store in numpy file for fast reading
#
#     """
#     np.random.seed(0)
#
#     #Get all samples
#     all_samples = loadMnist('train') + loadMnist('test')
#
#     #
#
#     #Empty arrays
#     train_data = np.zeros([28,28,10000])
#     train_label = np.zeros([10000])
#     val_data = np.zeros([28,28,2000])
#     val_label = np.zeros([2000])
#     test_data = np.zeros([28,28,50000])
#     test_label = np.zeros([50000])
#
#     i = 0
#     for j in range(10000):
#         sample =all_samples[i]
#         train_data[:, :, j] =  random_rotation(sample[0])
#         train_label[j] = sample[1]
#         i += 1
#pe(l):pe(l):
#     for j in range(2000):
#         sample = all_samples[i]
#         val_data[:, :, j] = random_rotation(sample[0])
#         val_label[j] = sample[1]
#         i += 1
#
#     for j in range(50000):
#         sample = all_samples[i]
#         test_data[:, :, j] = random_rotation(sample[0])
#         test_label[j] = sample[1]
#         i += 1
#
#
#     try:
#         os.mkdir('mnist_rot/')
#     except:
#         None
#     np.save('mnist_rot/train_data',train_data)
#     np.save('mnist_rot/train_label', train_label)
#     np.save('mnist_rot/val_data', val_data)
#     np.save('mnist_rot/val_label', val_label)
#     np.save('mnist_rot/test_data', test_data)
#     np.save('mnist_rot/test_label', test_label)
#
# if __name__ == '__main__':
#     makeMnistRot()
