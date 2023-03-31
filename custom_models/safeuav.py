import torch
from torch import nn, add
import torch.nn.functional as F


class Double(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same', bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.double_conv(x)
        return x1


class Down(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), bias=True, stride=(2, 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.down(x)
        return x1


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3),
                                             padding_mode='zeros', bias=True, stride=(2, 2))

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same', bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.conv_trans(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        concat = torch.cat([x2, x1], dim=1)
        return self.convs(concat)


def get_dilate(in_channels, out_channels, dilation, kernel_size=(3, 3)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', bias=True, dilation=dilation),
        nn.ReLU(inplace=True)
    )


class UNet_MDCB(nn.Module):
    def __init__(self, classes, in_channels=3, last='sigmoid'):
        super().__init__()
        init_nb = 24
        self.double1 = Double(in_channels, init_nb)
        self.down1 = Down(init_nb)
        self.double2 = Double(init_nb, init_nb * 2)
        self.down2 = Down(init_nb * 2)
        self.double3 = Double(init_nb * 2, init_nb * 4)
        self.down3 = Down(init_nb * 4)

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
        if last == 'sigmoid':
            self.last = nn.Sigmoid()
        elif last == 'softmax':
            self.last = nn.Softmax(dim=1)
        else:
            self.last = nn.Identity()

    def forward(self, x):
        s1 = self.double1(x)
        x1 = self.down1(s1)
        s2 = self.double2(x1)
        x2 = self.down2(s2)
        s3 = self.double3(x2)
        x3 = self.down3(s3)

        d1 = self.dilate1(x3)
        d2 = self.dilate2(d1)
        d3 = self.dilate3(d2)
        d4 = self.dilate4(d3)
        d5 = self.dilate5(d4)
        d6 = self.dilate6(d5)

        d_tot = add(add(add(add(add(d1, d2), d3), d4), d5), d6)

        up1 = self.up1(d_tot, s3)
        up2 = self.up2(up1, s2)
        up3 = self.up3(up2, s1)
        res = self.classify(up3)

        res = self.last(res)

        return res

