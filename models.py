import torch.nn as nn
from layers import maskAConv, MaskBConvBlock


class PixelCNN(nn.Module):
    def __init__(self, n_channel=1, h=128, layers = 1, feature_maps = 32, discrete_channel=256):
        """PixelCNN Model"""
        super(PixelCNN, self).__init__()

        self.discrete_channel = discrete_channel

        self.MaskAConv = maskAConv(n_channel, 2 * h, k_size=7, stride=1, pad=3)
        MaskBConv = []
        for i in range(layers):         
            MaskBConv.append(MaskBConvBlock(h, k_size=3, stride=1, pad=1))
        self.MaskBConv = nn.Sequential(*MaskBConv)

        # 1x1 conv to 3x256 channels    ( 3 = RGB; 1 if grayscale, 256 = 0-255)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(2 * h, feature_maps, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(),
            nn.Conv2d(feature_maps, n_channel * discrete_channel, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        """
        Args:
            x: [batch_size, channel, height, width]
        Return:
            out [batch_size, channel, height, width, 256]
        """
        batch_size, c_in, height, width = x.size()

        # [batch_size, 2h, 32, 32]
        x = self.MaskAConv(x)

        # [batch_size, 2h, 32, 32]
        x = self.MaskBConv(x)

        # [batch_size, 3x256, 32, 32]
        x = self.out(x)

        # [batch_size, channel, 256, 32, 32]
        x = x.view(batch_size, c_in, self.discrete_channel, height, width)

        # [batch_size, 256, channel, 32, 32]
        x = x.permute(0, 2, 1, 3, 4)

        return x
