# CMU CMU 18-780/6 Homework 4
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip
# CSC 321, Assignment 4
#
# This file contains the models used for both parts of the assignment:
#
#   - DCGenerator        --> Used in the vanilla GAN in Part 1
#   - DCDiscriminator    --> Used in both the vanilla GAN in Part 1
# For the assignment, you are asked to create the architectures of these
# three networks by filling in the __init__ and forward methods in the
# DCGenerator, DCDiscriminator classes.
# Feel free to add and try your own models

import torch
import torch.nn as nn

def Spectral_Normalization(W):

    # First we want to find the largest singular value of the matrix
    # Unlike the max of the matrix, we can think of this as the number
    # that represents the greatest possible "stretch" from a transformation
    # with an input vector. 

    # W can be thought of as the weights from a convolution
    # we want to reshape weight matrix such that the vertical
    # axis is the output channels, as we are concerned with the strech in the
    # output channels
    # below is deriving the largest singular value (from torch.linalg) and
    # normalizing the weight matrix with this value
    
    # ultimately, this limits the lipschitz constant which means greater model stability.
    # the Lipschitz contant controls for how fast a function can chnage. This is directly
    # affected by the largest singular value. Hence normalizing with largest 
    # singular value leads to greater model stability.

    # _, singular_values, _ = torch.linalg.svd(W)
    # S_largest = singular_values[0]

    W_shape = W.shape # get the full shape, not just the size of a single dimension

    W = W.view(W.size(0), -1)

    singular_vals = torch.linalg.svdvals(W)
    s_vals_max =  singular_vals[0]

    return (W/s_vals_max).view(W_shape)


def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
            scale_factor=2, norm='batch', activ=None):
    """Create a transposed-convolutional layer, with optional normalization."""
    layers = []
    layers.append(nn.Upsample(scale_factor=scale_factor, mode='nearest'))
    layers.append(nn.Conv2d(
        in_channels, out_channels,
        kernel_size, stride, padding, bias=norm is None
    ))
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
         norm='batch', init_zero_weights=False, activ=None):
    """Create a convolutional layer, with optional normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels, out_channels=out_channels,
        kernel_size=kernel_size, stride=stride, padding=padding,
        bias=norm is None
    )
    if init_zero_weights:
        conv_layer.weight.data = 0.001 * torch.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    if activ == 'relu':
        layers.append(nn.ReLU())
    elif activ == 'leaky':
        layers.append(nn.LeakyReLU())
    elif activ == 'tanh':
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class DCGenerator(nn.Module):

    def __init__(self, noise_size, conv_dim=64):
        super().__init__()

        ###########################################
        ##   FILL THIS IN: CREATE ARCHITECTURE   ##
        ###########################################

        # def up_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,
        #     scale_factor=2, norm='batch', activ=None):

        # norm = 'instance'
        # act = 'relu'

        self.up_conv1 = nn.Conv2d(noise_size, 256, 4, 1, 3) # produces a 4x4 from 1x1
        # self.up_conv1 = nn.Conv2d(100, 256, kernel_size=4, stride=1, padding=1)

        # critically, we don't use any padding to maximize learning from the raw input
        self.norm = nn.InstanceNorm2d(256)
        # self.norm = nn.InstanceNorm2d(256, eps=1e-3) 
        self.act1 = nn.ReLU()

        self.up_conv2 = up_conv(256, 128, 3, 1, 1, 2, norm = 'instance', activ='relu') # scale factor is already 2
        self.up_conv3 = up_conv(128, 64, 3, 1, 1, 2, norm = 'instance', activ='relu')
        self.up_conv4 = up_conv(64, 32, 3, 1, 1, 2, norm = 'instance', activ='relu')
        self.up_conv5 = up_conv(32, 3, 3, 1, 1, 2, norm = None, activ = 'tanh') # I wasn't sure if tanh was being applied, so I just added it as a function afterwards.
        # self.act2 = nn.Tanh()

    def forward(self, z):
        """
        Generate an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  16x100x1x1

        Output
        ------
            out: BS x channels x image_width x image_height  -->  16x3x64x64
        """
        # TODO
        # input ~ (16x100x1x1)
        # I have now added in the normalizations and activations.

        # we want to ensure that z is (16x100x1x1)
        z = z.view(z.size(0), 100, 1, 1)

        z = self.up_conv1(z)
        z = self.norm(z)
        z = self.act1(z)
        z = self.up_conv2(z)
        z = self.up_conv3(z)
        z = self.up_conv4(z)
        z = self.up_conv5(z) # should be 16 (batch size) x 3 x 64 x 64
        # z = self.act2(z)
        return z.squeeze()

class ResnetBlock(nn.Module):

    def __init__(self, conv_dim, norm, activ):
        super().__init__()
        self.conv_layer = conv(
            in_channels=conv_dim, out_channels=conv_dim,
            kernel_size=3, stride=1, padding=1, norm=norm,
            activ=activ
        )

    def forward(self, x):
        out = x + self.conv_layer(x)
        return out


class DCDiscriminator(nn.Module):
    """Architecture of the discriminator network."""

    def __init__(self, conv_dim=64):
        super().__init__()

        norm = None

        # def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
        #  norm='batch', init_zero_weights=False, activ=None):
        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'relu') # we can keep these all relatively the same
        
        self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'relu') # I double checked the math to make sure the feature

        self.conv3 = conv(64, 128, 4, 2, 1, norm, False, 'relu') # maps would half until the last convolution.

        self.conv4 = conv(128, 256, 4, 2, 1, norm, False, 'relu')

        self.conv5 = conv(256, 1, 4, 1, 0, norm = None, activ = None) # a stride of 1, no padding, and a kernel size of 4 fixes this problem
        # removed 'relu' from the last convolution layer, since it wasn't in the architecture.

    def forward(self, x): # I think this is already complete.
        """Forward pass, x is (B, C, H, W)."""

        # Similar to before, we only want to apply our normalization to the first four layers
        # not the last layer.

        # we use weight.data rather than just .weight to avoid breaking the gradients.
        
        # We ONLY want to apply spectral normaliation to the discriminator
        # the logic behind this is we want the discrimnator to struggle more to learn within a 
        # limited space (limited by the largest singular value), while the generator can
        # fluorish. Essentially, this prevents the discriminator from dominating.

        self.conv1[0].weight.data = Spectral_Normalization(self.conv1[0].weight.data)
        self.conv2[0].weight.data = Spectral_Normalization(self.conv2[0].weight.data)
        self.conv3[0].weight.data = Spectral_Normalization(self.conv3[0].weight.data)
        self.conv4[0].weight.data = Spectral_Normalization(self.conv4[0].weight.data)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.squeeze()
