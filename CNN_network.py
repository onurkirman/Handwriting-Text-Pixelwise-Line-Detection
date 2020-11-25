import torch
from torch.nn import Module, Sequential
from torch.nn import Conv2d, BatchNorm2d, ReLU, Dropout2d, MaxPool2d, UpsamplingNearest2d
from torchsummary import summary

class Network(Module):
    def __init__(self, number_of_classes, dropout_rate):
        super(Network, self).__init__()

        self.block1 = Sequential(
            Conv2d(1, 16, kernel_size=3, padding=1),
            BatchNorm2d(16),
            ReLU(),
            Dropout2d(dropout_rate),
            Conv2d(16, 16, kernel_size=3, padding=1),
            BatchNorm2d(16),
            ReLU()
        )

        self.pool1 = MaxPool2d((2,2))  # Reduces H,W by 2

        self.block2 = Sequential(
            Conv2d(16, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            Dropout2d(dropout_rate),
            Conv2d(32, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU()
        )
        
        self.pool2 = MaxPool2d((2,2))  # Reduces H,W by 2
        
        # Upsampling our predicted image from 64x64 to 256x256 (4 times)
        self.up1 = UpsamplingNearest2d(scale_factor=4)

        # Final reduce of channels to 0 & 1 (2 class)
        self.conv2d = Conv2d(32, number_of_classes, kernel_size=1)
        
        self.norm2d = BatchNorm2d(number_of_classes)

    def forward(self, x):
        out1 = self.block1(x)
        out_pool1 = self.pool1(out1)

        out2 = self.block2(out_pool1)
        out_pool2 = self.pool2(out2)

        out_up1 = self.up1(out_pool2)

        out3 = self.conv2d(out_up1)
        out_norm = self.norm2d(out3)

        y_pred = out_norm
        return y_pred

if __name__ == "__main__":
    number_of_classes = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Network(number_of_classes).to(device)
    summary(model, input_size=(1,256,256))
