from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class RecurrentBlock(nn.Module):
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv(x)
        for i in range(self.t):
            x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNNBlock, self).__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class SingleConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(AttUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        filters = 32
        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=filters)
        self.Conv2 = ConvBlock(ch_in=filters, ch_out=filters*2)
        self.Conv3 = ConvBlock(ch_in=filters*2, ch_out=filters*4)
        self.Conv4 = ConvBlock(ch_in=filters*4, ch_out=filters*8)
        self.Conv5 = ConvBlock(ch_in=filters*8, ch_out=filters*16)

        self.Up5 = UpConv(ch_in=filters*16, ch_out=filters*8)
        self.Att5 = AttentionBlock(F_g=filters*8, F_l=filters*8, F_int=filters*4)
        self.Up_conv5 = ConvBlock(ch_in=filters*16, ch_out=filters*8)

        self.Up4 = UpConv(ch_in=filters*8, ch_out=filters*4)
        self.Att4 = AttentionBlock(F_g=filters*4, F_l=filters*4, F_int=filters*2)
        self.Up_conv4 = ConvBlock(ch_in=filters*8, ch_out=filters*4)

        self.Up3 = UpConv(ch_in=filters*4, ch_out=filters*2)
        self.Att3 = AttentionBlock(F_g=filters*2, F_l=filters*2, F_int=filters)
        self.Up_conv3 = ConvBlock(ch_in=filters*4, ch_out=filters*2)

        self.Up2 = UpConv(ch_in=filters*2, ch_out=filters)
        self.Att2 = AttentionBlock(F_g=filters, F_l=filters, F_int=filters//2)
        self.Up_conv2 = ConvBlock(ch_in=filters*2, ch_out=filters)

        self.Conv_1x1 = nn.Conv2d(filters, output_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
