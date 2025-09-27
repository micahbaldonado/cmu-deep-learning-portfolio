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
import torch.nn.utils.spectral_norm as SN


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

    if norm == 'spectral':
        conv_layer = SN(conv_layer)

    layers.append(conv_layer)

    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channels))
    elif norm == 'instance':
        layers.append(nn.InstanceNorm2d(out_channels))

    # # ADDING SPECTRAL NORMALIZATION TO NORM CHANNEL
    # elif norm == 'spectral':
    #     layers.append(SN(conv_layer))

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



# NOW, THE DISCRIINATOR ACTS AS A CRITIC.
class DCDiscriminator(nn.Module):
    """Architecture of the discriminator network."""

    def __init__(self, conv_dim=64):
        super().__init__()

        norm = 'spectral'

        # def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,
        #  norm='batch', init_zero_weights=False, activ=None):

        # since this I am working with WGAN and not the spectral normalization GAN direclty, 
        # I will use the pytorch version of spectral normalization, since I am not trivializing the
        # process for Wasserrstein GAN.

        # The reason we want to apply spectral normalziation here is to ensure that the discriminator complies
        # with the 

        self.conv1 = conv(3, 32, 4, 2, 1, norm, False, 'relu') # we can keep these all relatively the same
        
        self.conv2 = conv(32, 64, 4, 2, 1, norm, False, 'relu') # I double checked the math to make sure the feature

        self.conv3 = conv(64, 128, 4, 2, 1, norm, False, 'relu') # maps would half until the last convolution.

        self.conv4 = conv(128, 256, 4, 2, 1, norm, False, 'relu')

        self.conv5 = conv(256, 1, 4, 1, 0, norm, activ = None) # a stride of 1, no padding, and a kernel size of 4 fixes this problem
        # removed 'relu' from the last convolution layer, since it wasn't in the architecture.

        # self.linear = nn.Linear(1, 1)

    def forward(self, x): # I think this is already complete.
        """Forward pass, x is (B, C, H, W)."""

        # Similar to before, we only want to apply our normalization to the first four layers
        # not the last layer.

        # we use weight.data rather than just .weight to avoid breaking the gradients.
        
        # We ONLY want to apply spectral normaliation to the discriminator
        # the logic behind this is we want the discrimnator to struggle more to learn within a 
        # limited space (limited by the largest singular value), while the generator can
        # fluorish. Essentially, this prevents the discriminator from dominating.
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.linear(x)

        return x.squeeze()
