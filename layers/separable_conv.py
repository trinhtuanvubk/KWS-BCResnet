from torch import nn

class DepthWiseConv(nn.Module):
    def __init__(self, in_size, out_size, stride=1, dilation=1):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_size,
                                    out_channels=out_size,
                                    kernel_size=(3,1),
                                    stride=stride,
                                    padding=(1,0),
                                    dilation=dilation,
                                    groups=in_size)
    def forward(self, x):
        x = self.depth_conv(x)
        return x


class DepthSeparableConv(nn.Module):
    def __init__(self, in_size, out_size, stride=1, dilation=1):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_size,
                                    out_channels=in_size,
                                    kernel_size=(1,3),
                                    stride=stride,
                                    padding=(0,1),
                                    dilation=dilation,
                                    groups=in_size)
        self.point_conv = nn.Conv2d(in_channels=in_size,
                                    out_channels=out_size,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x