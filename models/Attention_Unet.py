'''
Written by Onur Kirman Computer Science Undergrad at Ozyegin University
'''

import torch
from torch.nn import (BatchNorm2d, Conv2d, Dropout2d, LeakyReLU, MaxPool2d,
                      Module, ReLU, Sequential, Sigmoid, Upsample,
                      UpsamplingBilinear2d, UpsamplingNearest2d)
from torchsummary import summary


def conv_block(in_channel: int, out_channel: int, dropout_rate):
    conv_sequence = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            BatchNorm2d(out_channel),
            ReLU(inplace=True),
            Dropout2d(dropout_rate),
            Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            BatchNorm2d(out_channel),
            ReLU(inplace=True)
        )
    return conv_sequence


def up_conv(in_channel: int, out_channel: int):
    conv_seq = Sequential(
        Upsample(scale_factor=2),
        Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
        BatchNorm2d(out_channel),
        ReLU(inplace=True)
    )
    return conv_seq


def concatenate(out1, out2):
    return torch.cat((out1, out2), dim=1)


class Attention_block(Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = Sequential(
            Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            BatchNorm2d(F_int)
            )
        
        self.W_x = Sequential(
            Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            BatchNorm2d(F_int)
        )

        self.psi = Sequential(
            Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            BatchNorm2d(1),
            Sigmoid()
        )
        
        self.relu = ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Attention_Unet(Module):
    def __init__(self, number_of_classes, dropout_rate):
        super(Attention_Unet, self).__init__()

        self.block1 = conv_block(1, 32, dropout_rate=dropout_rate)

        self.pool1 = MaxPool2d((2, 2))

        self.block2 = conv_block(32, 64, dropout_rate=dropout_rate)

        self.pool2 = MaxPool2d((2, 2))

        self.block3 = conv_block(64, 128, dropout_rate=dropout_rate)

        self.pool3 = MaxPool2d((2, 2))

        self.block4 = conv_block(128, 256, dropout_rate=dropout_rate)

        self.pool4 = MaxPool2d((2, 2))




        self.block5 = conv_block(256, 512, dropout_rate=dropout_rate)




        self.up1 = up_conv(in_channel=512, out_channel=256)

        #here
        self.Att1 = Attention_block(F_g=256, F_l=256, F_int=128)

        self.block6 = conv_block(512, 256, dropout_rate=dropout_rate)




        self.up2 = up_conv(in_channel=256, out_channel=128)

        #here
        self.Att2 = Attention_block(F_g=128, F_l=128, F_int=64)

        self.block7 = conv_block(256, 128, dropout_rate=dropout_rate)




        self.up3 = up_conv(in_channel=128, out_channel=64)

        #here
        self.Att3 = Attention_block(F_g=64, F_l=64, F_int=32)

        self.block8 = conv_block(128, 64, dropout_rate=dropout_rate)




        self.up4 = up_conv(in_channel=64, out_channel=32)
        
        #here
        self.Att4 = Attention_block(F_g=32, F_l=32, F_int=16)

        self.block9 = conv_block(64, 32, dropout_rate=dropout_rate)



        # Single conv to #class
        self.conv2d = Conv2d(32, number_of_classes, kernel_size=1)
        self.norm2d = BatchNorm2d(number_of_classes)

    def forward(self, x):
        out1 = self.block1(x)               # Left Top of U

        out_pool1 = self.pool1(out1)

        out2 = self.block2(out_pool1)

        out_pool2 = self.pool2(out2)

        out3 = self.block3(out_pool2)

        out_pool3 = self.pool3(out3)

        out4 = self.block4(out_pool3)

        out_pool4 = self.pool4(out4)



        out5 = self.block5(out_pool4)       # Middle



        out_up1 = self.up1(out5)
        out4 = self.Att1(g=out_up1, x=out4)
        out6 = concatenate(out_up1, out4)
        out6 = self.block6(out6)





        out_up2 = self.up2(out6)
        out3 = self.Att2(g=out_up2, x=out3)
        out7 = concatenate(out_up2, out3)
        out7 = self.block7(out7)





        out_up3 = self.up3(out7)
        out2 = self.Att3(g=out_up3, x=out2)
        out8 = concatenate(out_up3, out2)
        out8 = self.block8(out8)





        out_up4 = self.up4(out8)
        out1 = self.Att4(g=out_up4, x=out1)
        out9 = concatenate(out_up4, out1)
        out9 = self.block9(out9)            # Left Top of U




        out10 = self.conv2d(out9)

        y_pred = self.norm2d(out10)

        return y_pred


if __name__ == '__main__':
    
    print(f'Cuda Available: {torch.cuda.is_available()}')
    print(f'{"Cuda Device Name: " + torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "No Cuda Device Found"}')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    number_of_classes = 2

    print(f'Number of Classes: {number_of_classes}')

    model = Attention_Unet(number_of_classes, 0).to(device)
    summary(model, input_size=(1, 256, 256))  # (channels, H, W)
