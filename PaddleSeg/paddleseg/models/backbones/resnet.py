# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn
from paddleseg.cvlibs import manager
from paddleseg.utils import utils

__all__ = ["ResNet50Encoder"]


class Layer1_0_downsample(nn.Layer):
    def __init__(self, conv0_out_channels, conv0_stride, conv0_in_channels, bn0_num_channels):
        super(Layer1_0_downsample, self).__init__()
        self.conv0 = nn.Conv2D(kernel_size=(1, 1), bias_attr=False, out_channels=conv0_out_channels,
                               stride=conv0_stride, in_channels=conv0_in_channels)
        self.bn0 = nn.BatchNorm2D(momentum=0.1, num_features=bn0_num_channels)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        return x2


class Bottleneck0(nn.Layer):
    def __init__(self, ):
        super(Bottleneck0, self).__init__()

        self.conv0 = nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.bn0 = nn.BatchNorm2D(num_features=128, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=128, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=128)
        self.bn1 = nn.BatchNorm2D(num_features=128, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=128)
        self.bn2 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck1(nn.Layer):
    def __init__(self, ):
        super(Bottleneck1, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.bn0 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=256, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=256)
        self.bn1 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn2 = nn.BatchNorm2D(num_features=1024, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck2(nn.Layer):
    def __init__(self, ):
        super(Bottleneck2, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.bn0 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=256, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=256)
        self.bn1 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn2 = nn.BatchNorm2D(num_features=1024, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck3(nn.Layer):
    def __init__(self, ):
        super(Bottleneck3, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.bn0 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=256, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=256)
        self.bn1 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn2 = nn.BatchNorm2D(num_features=1024, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck4(nn.Layer):
    def __init__(self, ):
        super(Bottleneck4, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.bn0 = nn.BatchNorm2D(num_features=128, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=128, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=128)
        self.bn1 = nn.BatchNorm2D(num_features=128, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=128)
        self.bn2 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck5(nn.Layer):
    def __init__(self, ):
        super(Bottleneck5, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=64, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn0 = nn.BatchNorm2D(num_features=64, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=64, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=64)
        self.bn1 = nn.BatchNorm2D(num_features=64, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.bn2 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck6(nn.Layer):
    def __init__(self, ):
        super(Bottleneck6, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.bn0 = nn.BatchNorm2D(num_features=128, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=128, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=128)
        self.bn1 = nn.BatchNorm2D(num_features=128, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=128)
        self.bn2 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck7(nn.Layer):
    def __init__(self, ):
        super(Bottleneck7, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.bn0 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=256, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=256)
        self.bn1 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn2 = nn.BatchNorm2D(num_features=1024, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck8(nn.Layer):
    def __init__(self, ):
        super(Bottleneck8, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.bn0 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=256, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=256)
        self.bn1 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn2 = nn.BatchNorm2D(num_features=1024, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck9(nn.Layer):
    def __init__(self, ):
        super(Bottleneck9, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=64, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.bn0 = nn.BatchNorm2D(num_features=64, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=64, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=64)
        self.bn1 = nn.BatchNorm2D(num_features=64, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.bn2 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.layer1_0_downsample0 = Layer1_0_downsample(conv0_out_channels=256, conv0_stride=[1, 1],
                                                        conv0_in_channels=64, bn0_num_channels=256)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.layer1_0_downsample0(x0)
        x10 = x8 + 1 * x9
        x11 = self.relu2(x10)
        return x11


class Bottleneck10(nn.Layer):
    def __init__(self, ):
        super(Bottleneck10, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=1024)
        self.bn0 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=512, kernel_size=(3, 3), bias_attr=False, stride=2, padding=1,
                               in_channels=512)
        self.bn1 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=2048, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.bn2 = nn.BatchNorm2D(num_features=2048, momentum=0.1)
        self.layer1_0_downsample0 = Layer1_0_downsample(conv0_out_channels=2048, conv0_stride=[2, 2],
                                                        conv0_in_channels=1024, bn0_num_channels=2048)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.layer1_0_downsample0(x0)
        x10 = x8 + 1 * x9
        x11 = self.relu2(x10)
        return x11


class Bottleneck11(nn.Layer):
    def __init__(self, ):
        super(Bottleneck11, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=128, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn0 = nn.BatchNorm2D(num_features=128, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=128, kernel_size=(3, 3), bias_attr=False, stride=2, padding=1,
                               in_channels=128)
        self.bn1 = nn.BatchNorm2D(num_features=128, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=128)
        self.bn2 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.layer1_0_downsample0 = Layer1_0_downsample(conv0_out_channels=512, conv0_stride=[2, 2],
                                                        conv0_in_channels=256, bn0_num_channels=512)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.layer1_0_downsample0(x0)
        x10 = x8 + 1 * x9
        x11 = self.relu2(x10)
        return x11


class Bottleneck12(nn.Layer):
    def __init__(self, ):
        super(Bottleneck12, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.bn0 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=256, kernel_size=(3, 3), bias_attr=False, stride=2, padding=1,
                               in_channels=256)
        self.bn1 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=1024, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn2 = nn.BatchNorm2D(num_features=1024, momentum=0.1)
        self.layer1_0_downsample0 = Layer1_0_downsample(conv0_out_channels=1024, conv0_stride=[2, 2],
                                                        conv0_in_channels=512, bn0_num_channels=1024)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = self.layer1_0_downsample0(x0)
        x10 = x8 + 1 * x9
        x11 = self.relu2(x10)
        return x11


class Bottleneck13(nn.Layer):
    def __init__(self, ):
        super(Bottleneck13, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=2048)
        self.bn0 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=512, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=512)
        self.bn1 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=2048, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.bn2 = nn.BatchNorm2D(num_features=2048, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck14(nn.Layer):
    def __init__(self, ):
        super(Bottleneck14, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=64, kernel_size=(1, 1), bias_attr=False, in_channels=256)
        self.bn0 = nn.BatchNorm2D(num_features=64, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=64, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=64)
        self.bn1 = nn.BatchNorm2D(num_features=64, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=256, kernel_size=(1, 1), bias_attr=False, in_channels=64)
        self.bn2 = nn.BatchNorm2D(num_features=256, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Bottleneck15(nn.Layer):
    def __init__(self, ):
        super(Bottleneck15, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=512, kernel_size=(1, 1), bias_attr=False, in_channels=2048)
        self.bn0 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2D(out_channels=512, kernel_size=(3, 3), bias_attr=False, padding=1, in_channels=512)
        self.bn1 = nn.BatchNorm2D(num_features=512, momentum=0.1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(out_channels=2048, kernel_size=(1, 1), bias_attr=False, in_channels=512)
        self.bn2 = nn.BatchNorm2D(num_features=2048, momentum=0.1)
        self.relu2 = nn.ReLU()

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x4 = self.conv1(x3)
        x5 = self.bn1(x4)
        x6 = self.relu1(x5)
        x7 = self.conv2(x6)
        x8 = self.bn2(x7)
        x9 = x8 + 1 * x0
        x10 = self.relu2(x9)
        return x10


class Layer1(nn.Layer):
    def __init__(self, ):
        super(Layer1, self).__init__()
        self.bottleneck90 = Bottleneck9()
        self.bottleneck50 = Bottleneck5()
        self.bottleneck140 = Bottleneck14()

    def forward(self, x0):
        x1 = self.bottleneck90(x0)
        x2 = self.bottleneck50(x1)
        x3 = self.bottleneck140(x2)
        return x3


class Layer2(nn.Layer):
    def __init__(self, ):
        super(Layer2, self).__init__()
        self.bottleneck110 = Bottleneck11()
        self.bottleneck40 = Bottleneck4()
        self.bottleneck60 = Bottleneck6()
        self.bottleneck00 = Bottleneck0()

    def forward(self, x0):
        x1 = self.bottleneck110(x0)
        x2 = self.bottleneck40(x1)
        x3 = self.bottleneck60(x2)
        x4 = self.bottleneck00(x3)
        return x4


class Layer3(nn.Layer):
    def __init__(self, ):
        super(Layer3, self).__init__()
        self.bottleneck120 = Bottleneck12()
        self.bottleneck10 = Bottleneck1()
        self.bottleneck70 = Bottleneck7()
        self.bottleneck30 = Bottleneck3()
        self.bottleneck80 = Bottleneck8()
        self.bottleneck20 = Bottleneck2()

    def forward(self, x0):
        x1 = self.bottleneck120(x0)
        x2 = self.bottleneck10(x1)
        x3 = self.bottleneck70(x2)
        x4 = self.bottleneck30(x3)
        x5 = self.bottleneck80(x4)
        x6 = self.bottleneck20(x5)
        return x6


class Layer4(nn.Layer):
    def __init__(self, ):
        super(Layer4, self).__init__()
        self.bottleneck100 = Bottleneck10()
        self.bottleneck150 = Bottleneck15()
        self.bottleneck130 = Bottleneck13()

    def forward(self, x0):
        x1 = self.bottleneck100(x0)
        x2 = self.bottleneck150(x1)
        x3 = self.bottleneck130(x2)
        return x3


class ResNet50(nn.Layer):
    def __init__(self, ):
        super(ResNet50, self).__init__()
        self.conv0 = nn.Conv2D(out_channels=64, kernel_size=(7, 7), bias_attr=False, stride=2, padding=3,
                               in_channels=3)
        self.bn0 = nn.BatchNorm2D(num_features=64, momentum=0.1)
        self.relu0 = nn.ReLU()
        self.maxpool0 = paddle.nn.MaxPool2D(kernel_size=[3, 3], stride=2, padding=1)
        self.layer10 = Layer1()
        self.layer20 = Layer2()
        self.layer30 = Layer3()
        self.layer40 = Layer4()
        self.x10 = paddle.nn.AdaptiveAvgPool2D(output_size=[1, 1])
        self.linear0 = paddle.nn.Linear(in_features=2048, out_features=1000)

    def forward(self, x0):
        x1 = self.conv0(x0)
        x2 = self.bn0(x1)
        x3 = self.relu0(x2)
        x5 = self.maxpool0(x3)
        x6 = self.layer10(x5)
        x7 = self.layer20(x6)
        x8 = self.layer30(x7)
        x9 = self.layer40(x8)
        x11 = self.x10(x9)
        x12 = paddle.flatten(x=x11, start_axis=1)
        x13 = self.linear0(x12)
        return x13


@manager.BACKBONES.add_component
class ResNet50Encoder(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 pretrained=None):
        super(ResNet50Encoder, self).__init__()

        self.resnet = ResNet50()
        if in_channels != 3:
            self.resnet.conv0 = nn.Conv2D(
                in_channels, 64, 7, 2, 3, bias_attr=False)

        if pretrained is not None:
            utils.load_pretrained_model(self.resnet, pretrained)

    def forward(self, x):
        x = self.resnet.conv0(x)
        x = self.resnet.bn0(x)
        x = self.resnet.relu0(x)
        x = self.resnet.maxpool0(x)

        c2 = self.resnet.layer10(x)
        c3 = self.resnet.layer20(c2)
        c4 = self.resnet.layer30(c3)
        c5 = self.resnet.layer40(c4)

        return [c2, c3, c4, c5]
