import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules import module
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DWConv(nn.Module):
    def __init__(self, in_channel, out_channel,k):
        super(DWConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=k, stride=1, padding=k//2, groups=in_channel)
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class GPSSpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(GPSSpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim, self.dim // reduction, kernel_size=1),  # 修改输入通道为dim
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 1, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, gm):
        x1 = gm*x1 + x1  # 特征增强,6456
        spatial_weights = self.mlp(x1)
        return spatial_weights
        #return spatial_weights * torch.sigmoid(gm) + spatial_weights#6420

class GPSChannelWeights2(nn.Module):
    def __init__(self, channel, subchannel, reduction):
        super(GPSChannelWeights2, self).__init__()
        self.dim = channel
        self.group = channel // subchannel
        valid_groups = [1, 2, 4, 8, 16, 32, 64]
        if self.group not in valid_groups:
            raise ValueError(f"Invalid group number: {self.group}")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channel * 2, channel * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel * 2 // reduction, channel * 2),
            nn.Sigmoid())
        self.mlp2 = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 3, padding=1),
            nn.Sigmoid()
        )
        self.DWCmlp = nn.Sequential(
            DWConv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            DWConv(channel //8, channel, 3),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1),
            nn.ReLU(True),
        )
        self.Dwconv = nn.Sequential(
            DWConv(channel + self.group, channel, 3),
            nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, gm):
        B, _, H, W = x.shape
        xs = torch.chunk(x, self.group, dim=1)
        interleaved = []
        for chunk in xs:
            interleaved.extend([chunk, gm])
        x_cat = torch.cat(interleaved, dim=1)
        xg = self.Dwconv(x_cat)

        return xg

class ReliabilityAwareFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.density_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, gps, mask):
        density_map = self.density_net(gps)
        return density_map * gps + (1 - density_map) * mask


class TMGM(nn.Module):
    def __init__(self, dim, reduction=8):
        super(TMGM, self).__init__()
        self.prconv = nn.Sequential(
            #nn.Conv2d(dim, 1, 1)
            nn.Conv2d(dim, dim//8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim//8, 1, 3, padding=1),

        )
        self.gmweight = ReliabilityAwareFusion(dim)
        self.spatial_weights = GPSSpatialWeights(dim=dim, reduction=reduction)
        self.channel_weights = GPSChannelWeights2(channel=dim, subchannel=24, reduction=reduction)#使用SwinT
        #self.channel_weights = GPSChannelWeights2(channel=dim, subchannel=16, reduction=reduction)

    def forward(self, x1, gps):
        mask = self.prconv(x1)
        gm = self.gmweight(gps, mask)
        channel_weights = self.channel_weights(x1, gm)
        x1_c = x1 + channel_weights
        spatial_weights = self.spatial_weights(x1_c, gm)
        x1_s = x1_c + spatial_weights * x1_c
        return x1_s

class Asterisk(nn.Module):
    def __init__(self, in_channels, kernel, dilation):
        super(Asterisk, self).__init__()
        outc = in_channels//4
        self.kernel = kernel
        self.dilation = dilation

        # 计算各方向的padding (确保输出尺寸不变)
        pad_h = (dilation * (kernel - 1)) // 2  # 垂直方向padding
        pad_w = (dilation * (kernel - 1)) // 2  # 水平方向padding

        # 四个分支的卷积定义
        # 分支1: 水平方向卷积 (1, kernel_w)
        self.conv_h = nn.Conv2d(
            in_channels, outc,
            kernel_size=(1, kernel),
            padding=(0, pad_w),
            dilation=(1, dilation))

        # 分支2: 垂直方向卷积 (kernel_h, 1)
        self.conv_v = nn.Conv2d(
            in_channels, outc,
            kernel_size=(kernel, 1),
            padding=(pad_h, 0),
            dilation=(dilation, 1))

        # 分支3: 对角线1 (左上↘右下)
        self.conv_diag1 = nn.Conv2d(
            in_channels, outc,
            kernel_size=(kernel, 1),
            padding=(pad_h, 0),
            dilation=(dilation, 1))

        # 分支4: 对角线2 (右上↙左下)
        self.conv_diag2 = nn.Conv2d(
            in_channels, outc,
            kernel_size=(1, kernel),
            padding=(0, pad_w),
            dilation=(1, dilation))

    def forward(self, x):
        # 分支1: 水平方向
        x1 = self.conv_h(x)

        # 分支2: 垂直方向
        x2 = self.conv_v(x)

        # 分支3: 对角线1 (需要空间变换)
        x3 = self.inv_h_transform(
            self.conv_diag1(self.h_transform(x))
        )

        # 分支4: 对角线2 (需要空间变换)
        x4 = self.inv_v_transform(
            self.conv_diag2(self.v_transform(x))
        )

        # 拼接所有分支
        return torch.cat([x1, x2, x3, x4], dim=1)
    def h_transform(self, x):
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x

    def inv_h_transform(self, x):
            shape = x.size()
            x = x.reshape(shape[0], shape[1], -1).contiguous()
            x = torch.nn.functional.pad(x, (0, shape[-2]))
            x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
            x = x[..., 0: shape[-2]]
            return x

    def v_transform(self, x):
            x = x.permute(0, 1, 3, 2)
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
            x = x.permute(0, 1, 3, 2)
            shape = x.size()
            x = x.reshape(shape[0], shape[1], -1)
            x = torch.nn.functional.pad(x, (0, shape[-2]))
            x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
            x = x[..., 0: shape[-2]]
            return x.permute(0, 1, 3, 2)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MCAP(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,3,5,7]):
        super(MCAP, self).__init__()
        group_size = dim_xh  // 2
        self.c0 = nn.Conv2d(group_size , group_size//2, 1)
        self.c1 = nn.Conv2d(group_size , group_size//2, 1)
        self.c2 = nn.Conv2d(group_size , group_size//2, 1)
        self.c3 = nn.Conv2d(group_size , group_size//2, 1)

        self.lsp0 = Asterisk(group_size//2,5,1)
        self.lsp1 = Asterisk(group_size//2, 9, 1)
        self.lsp2 = Asterisk(group_size//2, 9, 2)
        self.lsp3 = Asterisk(group_size//2, 9, 4)
        self.tail_conv = nn.Sequential(
            nn.Conv2d(dim_xh , dim_xh, 1),
            nn.ReLU()
        )

    def forward(self, xh, xl):
        xy = xh
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = torch.cat((xh[0], xl[0]), dim=1)
        x0 = self.c0(x0)
        x0 = self.lsp0(x0)

        x1 = torch.cat((xh[1], xl[1]), dim=1)
        x1 = self.c1(x1)
        x1 = self.lsp1(x1)

        x2 = torch.cat((xh[2], xl[2]), dim=1)
        x2 = self.c2(x2)
        x2 = self.lsp2(x2)

        x3 = torch.cat((xh[3], xl[3]), dim=1)
        x3 = self.c3(x3)
        x3 = self.lsp3(x3)

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x+xy


class MyDecoderHead(nn.Module): #输入4
    def __init__(self,
                 filters1,
                 num_classes=1,
                 dropout_ratio=0,
                 embed_dim=512,
                 ):

        super(MyDecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.filters1 = filters1


        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.downsample0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.tmgm4 = TMGM(filters1[2])
        self.tmgm3 = TMGM(filters1[1])
        self.tmgm2 = TMGM(filters1[0])
        self.tmgm1 = TMGM(filters1[0]//2)

        self.mcap4 = MCAP(filters1[2],filters1[2])
        self.mcap3 = MCAP(filters1[1], filters1[1])
        self.mcap2 = MCAP(filters1[0], filters1[0])


        self.decoder4 = DecoderBlock(filters1[3], filters1[2])
        self.decoder3 = DecoderBlock(filters1[2], filters1[1])
        self.decoder2 = DecoderBlock(filters1[1], filters1[0])
        self.decoder1 = DecoderBlock(filters1[0], filters1[0]//2)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(filters1[0]//2, filters1[0]//4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(filters1[0]//4),
            nn.ReLU()
        )
        self.finalconv1 = nn.Conv2d(filters1[0]//4, 1, 3, padding=1)


    def forward(self, x1,gps):

        gpsd0 = self.downsample0(gps)
        gpsd1 = self.downsample0(gpsd0)
        gpsd2 = self.downsample0(gpsd1)
        gpsd3 = self.downsample0(gpsd2)

        c1, c2, c3, c4 = x1


        d4 = self.decoder4(c4)
        d4 = self.tmgm4(d4, gpsd3)
        d4 = self.mcap4(d4, c3)

        d3 = self.decoder3(d4)
        d3 = self.tmgm3(d3, gpsd2)
        d3 = self.mcap3(d3, c2)

        d2 = self.decoder2(d3)
        d2 = self.tmgm2(d2, gpsd1)
        d2 = self.mcap2(d2, c1)

        d1 = self.decoder1(d2)
        d1 = self.tmgm1(d1, gpsd0)

        x = self.deconv1(d1)
        x = self.finalconv1(x)

        return torch.sigmoid(x)