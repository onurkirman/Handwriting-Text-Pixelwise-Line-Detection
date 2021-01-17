'''
Written by Onur Kirman Computer Science Undergrad at Ozyegin University
This script is the clipped version of Unet Model referenced.
Ref: U-Net: Convolutional Networks for Biomedical Image Segmentation [https://arxiv.org/abs/1505.04597]

Notes about network hyperparameters & layers:

- 'MaxPool2d' with (2,2) shrinks size by half in both dimensions.
-  UpsamplingNearest2d scale_factor of 2, just doubles the size.
- 'padding = 1' -> helps to keep the size of our input while conv.(3,3).

Conclutions over the Tests being done: 
- 'ReLu' gives better results than 'LeakyReLU'.
- 'Dropout' decreases accuracy on small datasets, like ours. 
- 'BatchNorm' increses accuracy. Also, it helps the neurons to better understand and converge.
- 'UpsamplingNearest2d' is better than 'UpsamplingBilinear2d' in pixelwise line seperations. 
'''

import torch
from torch.nn import (BatchNorm2d, Conv2d, Dropout2d, LeakyReLU, MaxPool2d,
                      Module, ReLU, Sequential, UpsamplingBilinear2d,
                      UpsamplingNearest2d)
from torchsummary import summary


def conv_block(in_channel: int, out_channel: int, dropout_rate):
    conv_sequence = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            BatchNorm2d(out_channel),
            ReLU(),
            Dropout2d(dropout_rate),
            Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            BatchNorm2d(out_channel),
            ReLU()
        )
    return conv_sequence


def concatenate(out1, out2):
    return torch.cat((out1, out2), dim=1)


class UnetModelClipped(Module):
    def __init__(self, number_of_classes, dropout_rate):
        super(UnetModelClipped, self).__init__()

        self.block1 = conv_block(1, 32, dropout_rate=dropout_rate)

        self.pool1 = MaxPool2d((2, 2))  # shrinks size by half

        self.block2 = conv_block(32, 64, dropout_rate=dropout_rate)

        self.pool2 = MaxPool2d((2, 2))  # shrinks size by half

        self.block3 = conv_block(64, 128, dropout_rate=dropout_rate)

        self.up1 = UpsamplingNearest2d(scale_factor=2)  # doubles the size

        self.block4 = conv_block(128 + 64, 64, dropout_rate=dropout_rate)

        self.up2 = UpsamplingNearest2d(scale_factor=2)  # doubles the size

        self.block5 = conv_block(64 + 32, 32, dropout_rate=dropout_rate)

        # Single conv to #class
        self.conv2d = Conv2d(32, number_of_classes, kernel_size=1)
        self.norm2d = BatchNorm2d(number_of_classes)

    def forward(self, x):
        out1 = self.block1(x)

        out_pool1 = self.pool1(out1)

        out2 = self.block2(out_pool1)

        out_pool2 = self.pool2(out2)

        out3 = self.block3(out_pool2)

        out_up1 = self.up1(out3)

        out4 = concatenate(out_up1, out2)
        out4 = self.block4(out4)

        out_up2 = self.up2(out4)

        out5 = concatenate(out_up2, out1)
        out5 = self.block5(out5)

        out6 = self.conv2d(out5)

        y_pred = self.norm2d(out6)

        return y_pred


if __name__ == '__main__':
    
    print(f'Cuda Available: {torch.cuda.is_available()}')
    print(f'{"Cuda Device Name: " + torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No Cuda Device Found"}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    number_of_classes = 2

    print(f'Number of Classes: {number_of_classes}')

    model = UnetModelClipped(number_of_classes, 0).to(device)
    summary(model, input_size=(1, 256, 256))  # (channels, H, W)
