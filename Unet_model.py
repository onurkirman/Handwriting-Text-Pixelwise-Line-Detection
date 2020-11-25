import torch
from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, ReLU, LeakyReLU, Dropout2d, MaxPool2d, UpsamplingNearest2d, UpsamplingBilinear2d
from torchsummary import summary

'''
Notes about network hyperparameters & layers:

- 'MaxPool2d' with (2,2) makes the size half in both dim.
- 'padding = 1' -> helps to keep the size of our input while conv.(3,3). It is same as " padding='same' " in keras

- 'Dropout' decreases acc.
- 'ReLu' gives better results than 'LeakyReLU'
- 'BatchNorm' increses acc. and helps better understand the neurons to converge
- 'UpsamplingNearest2d' > 'UpsamplingBilinear2d' in line seperations

'''
class UnetModel(Module):
    def __init__(self, number_of_classes, dropout_rate):
        super(UnetModel, self).__init__()
        
        k = 0
        
        self.block1 = Sequential(
            Conv2d(1, 32 + k, kernel_size=3, padding=1),
            BatchNorm2d(32 + k),
            ReLU(),
            Dropout2d(dropout_rate),
            Conv2d(32 + k, 32 + k, kernel_size=3, padding=1),
            BatchNorm2d(32 + k),
            ReLU(),
        )
        
        self.pool1 = MaxPool2d((2, 2))  # shrinks size by half

        self.block2 = Sequential(
            Conv2d(32 + k, 64 + k, kernel_size=3, padding=1),
            BatchNorm2d(64 + k),
            ReLU(),
            Dropout2d(dropout_rate),
            Conv2d(64 + k, 64 + k, kernel_size=3, padding=1),
            BatchNorm2d(64 + k),
            ReLU(),
        )
        
        self.pool2 = MaxPool2d((2, 2))  # shrinks size by half

        self.block3 = Sequential(
            Conv2d(64 + k, 128 + k, kernel_size=3, padding=1),
            BatchNorm2d(128 + k),
            ReLU(),
            Dropout2d(dropout_rate),
            Conv2d(128 + k, 128 + k, kernel_size=3, padding=1),
            BatchNorm2d(128 + k),
            ReLU()
        )

        self.up1 = UpsamplingNearest2d(scale_factor=2)  # doubles the size

        self.block4 = Sequential(
            Conv2d(128 + k  +  64 + k, 64 + k, kernel_size=3, padding=1),
            BatchNorm2d(64 + k),
            ReLU(),
            Dropout2d(dropout_rate),
            Conv2d(64 + k, 64 + k, kernel_size=3, padding=1),
            BatchNorm2d(64 + k),
            ReLU()
        )

        self.up2 = UpsamplingNearest2d(scale_factor=2)  # doubles the size
        
        self.block5 = Sequential(
            Conv2d(64 + k  +  32 + k, 32 + k, kernel_size=3, padding=1),
            BatchNorm2d(32 + k),
            ReLU(),
            Dropout2d(dropout_rate),
            Conv2d(32 + k, 32 + k, kernel_size=3, padding=1),
            BatchNorm2d(32 + k),
            ReLU()
        )

        self.conv2d = Conv2d(32 + k, number_of_classes, kernel_size=1)
        
        self.norm2d = BatchNorm2d(number_of_classes)

        # no LogSoftmax here. Because We only need 2 classes, it's used for 2+ classes and NLL_Loss

    def forward(self, x):
        out1 = self.block1(x)

        out_pool1 = self.pool1(out1)

        out2 = self.block2(out_pool1)
        
        out_pool2 = self.pool1(out2)

        out3 = self.block3(out_pool2)

        out_up1 = self.up1(out3)
        
        out4 = torch.cat((out_up1, out2), dim=1)
        out4 = self.block4(out4)

        out_up2 = self.up2(out4)
        
        out5 = torch.cat((out_up2, out1), dim=1)
        out5 = self.block5(out5)

        out6 = self.conv2d(out5)
        
        y_pred = self.norm2d(out6)

        return y_pred


if __name__ == '__main__':

    number_of_classes = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UnetModel(number_of_classes).to(device)
    summary(model, input_size=(1, 256, 256))  # (channels, H, W)

    