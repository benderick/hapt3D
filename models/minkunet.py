# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.

import torch
import torch.nn as nn
from torch.optim import SGD

import MinkowskiEngine as ME

from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from .resnet import ResNetBase


class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, instance_decoder=False, D=3, use_tanh=False):
        self.instance_decoder = instance_decoder
        self.use_tanh = use_tanh
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])
        self.inplanes_ins = self.inplanes
        
        # decoder
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])
        
        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])

        self.final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

        if self.instance_decoder:
            # instance decoder 1
            self.convtr4p16s2_ins1 = ME.MinkowskiConvolutionTranspose(
            self.inplanes_ins, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
            self.bntr4_ins1 = ME.MinkowskiBatchNorm(self.PLANES[4])

            inplanes_ins1 = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
            self.inplanes = inplanes_ins1
            self.block5_ins1 = self._make_layer(self.BLOCK, self.PLANES[4],
                                        self.LAYERS[4])
            self.convtr5p8s2_ins1 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
            self.bntr5_ins1 = ME.MinkowskiBatchNorm(self.PLANES[5])

            inplanes_ins1 = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
            self.inplanes = inplanes_ins1
            self.block6_ins1 = self._make_layer(self.BLOCK, self.PLANES[5],
                                        self.LAYERS[5])
            self.convtr6p4s2_ins1 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
            self.bntr6_ins1 = ME.MinkowskiBatchNorm(self.PLANES[6])

            inplanes_ins1 = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
            self.inplanes = inplanes_ins1
            self.block7_ins1 = self._make_layer(self.BLOCK, self.PLANES[6],
                                        self.LAYERS[6])
            self.convtr7p2s2_ins1 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
            self.bntr7_ins1 = ME.MinkowskiBatchNorm(self.PLANES[7])

            inplanes_ins1 = self.PLANES[7] + self.INIT_DIM
            self.inplanes = inplanes_ins1
            self.block8_ins1 = self._make_layer(self.BLOCK, self.PLANES[7],
                                        self.LAYERS[7])

            self.final_ins1 = ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion,
                3,  # originally out_channels, 3 is the offset dimension
                kernel_size=1,
                bias=True,
                dimension=D)

            # instance decoder 2
            self.convtr4p16s2_ins2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes_ins, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
            self.bntr4_ins2 = ME.MinkowskiBatchNorm(self.PLANES[4])

            inplanes_ins2 = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
            self.inplanes = inplanes_ins2
            self.block5_ins2 = self._make_layer(self.BLOCK, self.PLANES[4],
                                        self.LAYERS[4])
            self.convtr5p8s2_ins2 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
            self.bntr5_ins2 = ME.MinkowskiBatchNorm(self.PLANES[5])

            inplanes_ins2 = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
            self.inplanes = inplanes_ins2
            self.block6_ins2 = self._make_layer(self.BLOCK, self.PLANES[5],
                                        self.LAYERS[5])
            self.convtr6p4s2_ins2 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
            self.bntr6_ins2 = ME.MinkowskiBatchNorm(self.PLANES[6])

            inplanes_ins2 = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
            self.inplanes = inplanes_ins2
            self.block7_ins2 = self._make_layer(self.BLOCK, self.PLANES[6],
                                        self.LAYERS[6])
            self.convtr7p2s2_ins2 = ME.MinkowskiConvolutionTranspose(
                self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
            self.bntr7_ins2 = ME.MinkowskiBatchNorm(self.PLANES[7])

            inplanes_ins2 = self.PLANES[7] + self.INIT_DIM
            self.inplanes = inplanes_ins2
            self.block8_ins2 = self._make_layer(self.BLOCK, self.PLANES[7],
                                        self.LAYERS[7])

            self.final_ins2 = ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion,
                3,  # originally out_channels, 3 is the offset dimension
                kernel_size=1,
                bias=True,
                dimension=D)

    def forward(self, x):
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_encoder = self.block4(out)
        
        # tensor_stride=8
        out_sem = self.convtr4p16s2(out_encoder)
        out_sem = self.bntr4(out_sem)
        out_sem = self.relu(out_sem)

        out_sem = ME.cat(out_sem, out_b3p8)
        out_sem = self.block5(out_sem)

        # tensor_stride=4
        out_sem = self.convtr5p8s2(out_sem)
        out_sem = self.bntr5(out_sem)
        out_sem = self.relu(out_sem)

        out_sem = ME.cat(out_sem, out_b2p4)
        out_sem = self.block6(out_sem)

        # tensor_stride=2
        out_sem = self.convtr6p4s2(out_sem)
        out_sem = self.bntr6(out_sem)
        out_sem = self.relu(out_sem)

        out_sem = ME.cat(out_sem, out_b1p2)
        out_sem = self.block7(out_sem)

        # tensor_stride=1
        out_sem = self.convtr7p2s2(out_sem)
        out_sem = self.bntr7(out_sem)
        out_sem = self.relu(out_sem)

        out_sem = ME.cat(out_sem, out_p1)
        out_sem = self.block8(out_sem)

        out_ins1, out_ins2 = None, None 
        if self.instance_decoder:
            # decoder 1

            # tensor_stride=8
            out_ins1 = self.convtr4p16s2_ins1(out_encoder)
            out_ins1 = self.bntr4_ins1(out_ins1)
            out_ins1 = self.relu(out_ins1)

            out_ins1 = ME.cat(out_ins1, out_b3p8)
            out_ins1 = self.block5_ins1(out_ins1)

            # tensor_stride=4
            out_ins1 = self.convtr5p8s2_ins1(out_ins1)
            out_ins1 = self.bntr5_ins1(out_ins1)
            out_ins1 = self.relu(out_ins1)

            out_ins1 = ME.cat(out_ins1, out_b2p4)
            out_ins1 = self.block6_ins1(out_ins1)

            # tensor_stride=2
            out_ins1 = self.convtr6p4s2_ins1(out_ins1)
            out_ins1 = self.bntr6_ins1(out_ins1)
            out_ins1 = self.relu(out_ins1)

            out_ins1 = ME.cat(out_ins1, out_b1p2)
            out_ins1 = self.block7_ins1(out_ins1)

            # tensor_stride=1
            out_ins1 = self.convtr7p2s2_ins1(out_ins1)
            out_ins1 = self.bntr7_ins1(out_ins1)
            out_ins1 = self.relu(out_ins1)

            out_ins1 = ME.cat(out_ins1, out_p1)
            out_ins1 = self.block8_ins1(out_ins1)

            out_ins1 = self.final_ins1(out_ins1)
            if self.use_tanh: out_ins1 = ME.MinkowskiTanh()(out_ins1)

            # decoder 2
            
            # tensor_stride=8
            out_ins2 = self.convtr4p16s2_ins2(out_encoder)
            out_ins2 = self.bntr4_ins2(out_ins2)
            out_ins2 = self.relu(out_ins2)

            out_ins2 = ME.cat(out_ins2, out_b3p8)
            out_ins2 = self.block5_ins2(out_ins2)

            # tensor_stride=4
            out_ins2 = self.convtr5p8s2_ins2(out_ins2)
            out_ins2 = self.bntr5_ins2(out_ins2)
            out_ins2 = self.relu(out_ins2)

            out_ins2 = ME.cat(out_ins2, out_b2p4)
            out_ins2 = self.block6_ins2(out_ins2)

            # tensor_stride=2
            out_ins2 = self.convtr6p4s2_ins2(out_ins2)
            out_ins2 = self.bntr6_ins2(out_ins2)
            out_ins2 = self.relu(out_ins2)

            out_ins2 = ME.cat(out_ins2, out_b1p2)
            out_ins2 = self.block7_ins2(out_ins2)

            # tensor_stride=1
            out_ins2 = self.convtr7p2s2_ins2(out_ins2)
            out_ins2 = self.bntr7_ins2(out_ins2)
            out_ins2 = self.relu(out_ins2)

            out_ins2 = ME.cat(out_ins2, out_p1)
            out_ins2 = self.block8_ins2(out_ins2)

            out_ins2 = self.final_ins2(out_ins2)
            if self.use_tanh: out_ins2 = ME.MinkowskiTanh()(out_ins2)

        return self.final(out_sem), out_ins1, out_ins2


class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)


if __name__ == '__main__':
    from tests.python.common import data_loader
    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = MinkUNet14A(in_channels=3, out_channels=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    optimizer = SGD(net.parameters(), lr=1e-2)

    for i in range(10):
        optimizer.zero_grad()

        # Get new data
        coords, feat, label = data_loader(is_classification=False)
        input = ME.SparseTensor(feat, coordinates=coords, device=device)
        label = label.to(device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, label)
        print('Iteration: ', i, ', Loss: ', loss.item())

        # Gradient
        loss.backward()
        optimizer.step()

    # Saving and loading a network
    torch.save(net.state_dict(), 'test.pth')
    net.load_state_dict(torch.load('test.pth'))