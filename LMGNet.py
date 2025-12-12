import torch.nn as nn
import torch
from thop import profile
from networks.LMGNet.swin_transformer import SwinTransformer
from networks.LMGNet.DecoderHead import MyDecoderHead

class LMGNet(nn.Module):
    def __init__(self):
        super(LMGNet, self).__init__()

        filters = [96, 192, 384, 768]
        self.backboon = SwinTransformer()
        self.backboon.init_weights(r"D:\CV\CMIPNet-main\weights\swin_tiny_patch4_window7_224.pth")
        self.mdecoder = MyDecoderHead(filters1=filters)

    def forward(self,x):
        x1 = x[:, :3, :, :]  # image
        x2 = x[:, 3:, :, :]  # gps_map or lidar_map

        x1 = self.backboon(x1)

        x = self.mdecoder(x1,x2)
        return x

if __name__ == "__main__":
    model = LMGNet()
    input = torch.rand(1, 4, 512, 512)
    y = model(input)
    print(y.shape)
    flops, params = profile(model, (input,))
    print('flops: %.2f G, params: %.2f M' % (flops / 1000000.0 / 1024.0, params / 1000000.0))
    print(f"输出形状：{y.shape}")  # 应为[2,1,512,512]
    print(f"数值范围：[{y.min():.3f},  {y.max():.3f}]")  # 应在0-1之间
