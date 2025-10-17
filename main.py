import torch
import torch.nn as nn

# --- YOLOv8 Backbone ---
class Conv(nn.Module):
    """Standard Convolution with BatchNorm and SiLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    """Faster and Lighter Bottleneck"""
    def __init__(self, in_channels, out_channels, repeats=1):
        super().__init__()
        layers = []
        for _ in range(repeats):
            layers.append(Conv(in_channels, out_channels))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.conv = Conv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        return self.conv(torch.cat([x, self.maxpool(x)] * 3, dim=1))

# --- LFAM and MUF-Net Modules ---
class LFAM(nn.Module):
    """Lightweight Feature Aggregation Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = Conv(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)

class MFM(nn.Module):
    """Multi-Feature Fusion Module"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv1x1 = Conv(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x1, x2, x3):
        fused = torch.cat([x1, x2, x3], dim=1)
        return self.conv1x1(fused)

class MUFNet(nn.Module):
    """MUF-Net for feature fusion"""
    def __init__(self, in_channels):
        super().__init__()
        self.mfm = MFM(in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x1, x2, x3):
        fused = self.mfm(x1, x2, x3)
        return self.upsample(fused)

# --- YOLOv8 Head ---
class Detect(nn.Module):
    """Detection Head"""
    def __init__(self, nc=80, anchors=None):
        super().__init__()
        self.nc = nc  # number of classes
        self.anchors = anchors
        self.stride = None

    def forward(self, x):
        outputs = []
        for feat in x:
            outputs.append(feat)
        return outputs

# --- Main Network ---
class YOLOv8sCustom(nn.Module):
    """YOLOv8s with LFAM and MUF-Net"""
    def __init__(self, nc=80, depth_multiple=0.33, width_multiple=0.5):
        super().__init__()
        self.nc = nc
        self.depth_multiple = depth_multiple
        self.width_multiple = width_multiple

        # Backbone
        self.backbone = nn.Sequential(
            Conv(3, 64, kernel_size=3, stride=2),  # P1/2
            Conv(64, 128, kernel_size=3, stride=2),  # P2/4
            C2f(128, 128, repeats=3),
            Conv(128, 256, kernel_size=3, stride=2),  # P3/8
            C2f(256, 256, repeats=6),
            Conv(256, 512, kernel_size=3, stride=2),  # P4/16
            C2f(512, 512, repeats=6),
            Conv(512, 1024, kernel_size=3, stride=2),  # P5/32
            C2f(1024, 1024, repeats=3),
            SPPF(1024, 1024),
        )

        # LFAM Modules
        self.lfam = LFAM(256)

        # MUF-Net Modules
        self.muf_net = MUFNet(256)

        # Head
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            MFM(256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            MFM(256),
            Conv(256, 256, kernel_size=3, stride=2),
            MFM(256),
            Conv(256, 512, kernel_size=3, stride=2),
            MFM(512),
            Detect(nc=nc)
        )

    def forward(self, x):
        # Backbone
        feats = []
        for layer in self.backbone:
            x = layer(x)
            feats.append(x)

        # LFAM
        m3 = self.lfam(feats[-3])  # P3/8
        m4 = self.lfam(feats[-2])  # P4/16
        m5 = self.lfam(feats[-1])  # P5/32

        # MUF-Net
        fused = self.muf_net(m3, m4, m5)

        # Head
        return self.head(fused)

# Example usage
if __name__ == "__main__":
    model = YOLOv8sCustom(nc=80)
    input_tensor = torch.randn(1, 3, 640, 640)  # Input: B, C, H, W
    output = model(input_tensor)
    print("Output length:", len(output))
    for feat in output:
        print(feat.shape)