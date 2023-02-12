import torch
from torch import nn, add


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding_mode='same', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding_mode='same', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding_mode='same', bias=True, stride=(2, 2)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3),
                                             padding_mode='same', bias=True, stride=(2, 2))

        self.convs = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding_mode='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding_mode='same', bias=True),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = self.conv_trans(x1)
        concat = torch.cat([x2, x1])

        return self.convs(concat)


def get_dilate(in_channels, out_channels, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding_mode='same', bias=True, dilation=dilation),
        nn.ReLU(inplace=True)
    )


class UNet_MDCB(nn.Module):
    def __init__(self, init_nb, classes):
        super().__init__()
        self.down1 = DoubleConv(3, init_nb)
        self.down2 = DoubleConv(init_nb, init_nb * 2)
        self.down2 = DoubleConv(init_nb * 2, init_nb * 4)

        self.dilate1 = get_dilate(init_nb * 4, init_nb * 8, dilation=(1, 1))
        self.dilate2 = get_dilate(init_nb * 8, init_nb * 8, dilation=(2, 2))
        self.dilate3 = get_dilate(init_nb * 8, init_nb * 8, dilation=(4, 4))
        self.dilate4 = get_dilate(init_nb * 8, init_nb * 8, dilation=(8, 8))
        self.dilate5 = get_dilate(init_nb * 8, init_nb * 8, dilation=(16, 16))
        self.dilate6 = get_dilate(init_nb * 8, init_nb * 8, dilation=(32, 32))

        self.up1 = Up(init_nb * 8, init_nb * 4)
        self.up2 = Up(init_nb * 4, init_nb * 2)
        self.up3 = Up(init_nb * 2, init_nb)

        self.classify = nn.Conv2d(init_nb, classes, kernel_size=(1, 1), bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        d1 = self.dilate1(x3)
        d2 = self.dilate2(d1)
        d3 = self.dilate3(d2)
        d4 = self.dilate4(d3)
        d5 = self.dilate5(d4)
        d6 = self.dilate6(d5)

        d_tot = add(add(add(add(add(d1, d2), d3), d4), d5), d6)

        up1 = self.up1(d_tot, x3)
        up2 = self.up2(up1, x2)
        up3 = self.up3(up2, x1)
        res = self.classify(up3)

        return res
