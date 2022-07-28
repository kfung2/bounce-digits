import torch.nn as nn
import pytorch_lightning as pl

class Conv3dBlock(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)):
        super().__init__()

        self.mod = nn.Sequential(
            nn.Conv3d(input_channels, output_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm3d(self.output_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.mod(x)

class Conv3dDownsample(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        output_channels = input_channels * 2

        self.mod = nn.Sequential(
            nn.Conv3d(input_channels, output_channels,
                      stride=(1, 2, 2),
                      kernel_size=(3,3,3), padding=(1,1,1)),
            nn.BatchNorm3d(self.output_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.mod(x)

class ConvTranspose3dBlock(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1)):
        super().__init__()

        self.mod = nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels,
                               kernel_size=kernel_size, padding=padding, 
                               stride=stride),
            nn.BatchNorm3d(self.output_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.mod(x)

class ConvTranspose3dUpsample(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        output_channels = input_channels // 2

        self.mod = nn.Sequential(
            nn.ConvTranspose3d(input_channels, output_channels,
                               stride=(1,2,2),kernel_size=(3,3,3), padding=(1,1,1), output_padding=(0,1,1)),
            nn.BatchNorm3d(self.output_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.mod(x)

class ThreeDConvWideTwoDeepTwo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.mod = nn.Sequential(
            Conv3dBlock(1, 8), 
            Conv3dBlock(8, 8), Conv3dBlock(8, 8), 
            Conv3dDownsample(8),
            Conv3dBlock(16, 16), Conv3dBlock(16,16),
            Conv3dDownsample(16),
            Conv3dBlock(32, 32), Conv3dBlock(32, 32),
            ConvTranspose3dBlock(32, 32), ConvTranspose3dBlock(32, 32),
            ConvTranspose3dUpsample(32),
            ConvTranspose3dBlock(16, 16), ConvTranspose3dBlock(16, 16),
            ConvTranspose3dUpsample(16),
            ConvTranspose3dBlock(8, 8), ConvTranspose3dBlock(8, 8),
            nn.Conv3d(8, 1, kernel_size = (1,1,1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mod(x)

