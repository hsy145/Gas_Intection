"""
模型定义模块
包含 UNet、UNet-GCM、YOLOv5-Seg、EAEFNet、Segformer、PSPNet、MFNet、RTFNet、FEANet 等分割模型架构

可用模型:
- unet: 标准 UNet
- unet_gcm: UNet + 全局上下文模块
- yolov5seg: YOLOv5 分割网络
- eaefnet: 基于 FPN 的轻量级网络 (MobileNetV2)
- segformer: Transformer 分割网络 (MIT)
- pspnet: 金字塔场景解析网络 (PSPNet)
- mfnet: 多光谱融合网络（轻量级，参数量小）
- rtfnet: RGB-Thermal 融合网络（深层编码器）
- feanet: 特征增强注意力网络（IROS 2021）
"""
from typing import Dict, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """双卷积块：Conv-BN-ReLU-Conv-BN-ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ChannelAttentionModule(nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


import torch
import torch.nn as nn


class SkipAttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, reduction: int = 16):
        super().__init__()

        # 1. 双路池化 (Dual Pooling)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 2. 共享感知机 (Shared MLP)
        mid_channels = max(1, skip_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(skip_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, skip_channels, 1, bias=False)
        )

        # 3. 激活函数 - 对应图4的 Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip_feat: torch.Tensor, gate_feat: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            skip_feat: Encoder传来的特征
            gate_feat: Decoder传来的特征
        """

        # 1. 分别通过 AvgPool 和 MaxPool
        avg_out = self.avg_pool(skip_feat)
        max_out = self.max_pool(skip_feat)

        # 2. 通过共享 MLP (Shared MLP)
        # 注意：两个输出分别通过同一个 MLP
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        # 3. 融合
        out = avg_out + max_out

        # 4. 生成权重 (Sigmoid)
        weight = self.sigmoid(out)

        # 5. 重校准 (Element-wise Multiply) - 对应图4最右侧的乘号
        # 利用广播机制，将 (B, C, 1, 1) 的权重乘回 (B, C, H, W) 的特征
        refined = skip_feat * weight

        return refined


class GCM(nn.Module):
    """全局上下文模块 (Global Context Module)
    
    利用多尺度卷积核 (1×1, 1×3/3×1, 1×5/5×1, 1×7/7×1) 聚合不同感受视野，
    再通过通道注意力增强特征表达能力。
    """
    def __init__(self, in_channels, out_channels=None, reduction=16):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        # 四个多尺度分支
        # 分支1: 1×1 卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 分支2: 1×3 + 3×1 卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 分支3: 1×5 + 5×1 卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 分支4: 1×7 + 7×1 卷积
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 通道注意力
        self.channel_attention = ChannelAttentionModule(out_channels, reduction=reduction)
        
    def forward(self, x):
        # 四个分支并行处理
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        
        # 拼接多尺度特征
        out = torch.cat([b1, b2, b3, b4], dim=1)
        
        # 通道注意力增强
        out = self.channel_attention.forward(out)

        return out


class UNet(nn.Module):
    """标准 UNet 分割网络"""
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1.forward(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2.forward(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3.forward(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4.forward(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3.forward(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2.forward(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1.forward(x)

        out = self.conv_last(x)
        return out


class UNetGCM(nn.Module):
    """UNet with Global Context Module and decoder-guided skip attention.

    Args:
        in_ch: 输入通道数。
        out_ch: 输出通道数。
        use_gcm: 是否在瓶颈层启用 GCM 模块（消融用）。
        use_skip_attention: 是否启用 SkipAttentionGate（消融用）。
        gcm_reduction: GCM 内部通道注意力的 reduction 系数。
    """

    def __init__(self, in_ch=1, out_ch=1, *, use_gcm: bool = True,
                 use_skip_attention: bool = True, gcm_reduction: int = 16):
        super().__init__()
        self.use_gcm = use_gcm
        self.use_skip_attention = use_skip_attention

        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        
        # GCM 模块应用于最高层特征
        self.gcm = GCM(512, 512, reduction=gcm_reduction) if use_gcm else nn.Identity()

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.skip_attn3 = SkipAttentionGate(skip_channels=256, gate_channels=256) if use_skip_attention else None
        self.skip_attn2 = SkipAttentionGate(skip_channels=128, gate_channels=128) if use_skip_attention else None
        self.skip_attn1 = SkipAttentionGate(skip_channels=64, gate_channels=64) if use_skip_attention else None

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = DoubleConv(256 + 256, 256)
        self.dconv_up2 = DoubleConv(128 + 128, 128)
        self.dconv_up1 = DoubleConv(64 + 64, 64)

        self.conv_last = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1.forward(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2.forward(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3.forward(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4.forward(x)
        
        # 应用 GCM 模块增强高层语义特征
        x = self.gcm.forward(x)

        x = self.up3(x)
        filtered_conv3 = self.skip_attn3.forward(conv3, x)
        x = torch.cat([x, filtered_conv3], dim=1)
        x = self.dconv_up3.forward(x)

        x = self.up2(x)
        filtered_conv2 = self._filter_skip(conv2, x, self.skip_attn2)
        x = torch.cat([x, filtered_conv2], dim=1)
        x = self.dconv_up2.forward(x)

        x = self.up1(x)
        filtered_conv1 = self._filter_skip(conv1, x, self.skip_attn1)
        x = torch.cat([x, filtered_conv1], dim=1)
        x = self.dconv_up1.forward(x)

        out = self.conv_last(x)
        return out

    @staticmethod
    def _filter_skip(skip_feat: torch.Tensor, gate_feat: torch.Tensor,
                     gate_module: Optional[SkipAttentionGate]) -> torch.Tensor:
        if gate_module is None:
            return skip_feat
        return gate_module.forward(skip_feat, gate_feat)


class UNetGCMOld(nn.Module):
    """
    UNet + GCM (Global Context Module)
    """
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)
        
        # GCM 模块应用于最高层特征
        self.gcm = GCM(512, 512, reduction=16)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.dconv_down1.forward(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2.forward(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3.forward(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4.forward(x)
        
        # 应用 GCM 模块增强高层语义特征
        x = self.gcm.forward(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3.forward(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2.forward(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1.forward(x)

        out = self.conv_last(x)
        return out


class Conv(nn.Module):
    """标准卷积块：Conv + BN + SiLU"""
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, in_ch, out_ch, n=1, shortcut=True):
        super().__init__()
        hidden_ch = out_ch // 2
        self.cv1 = Conv(in_ch, hidden_ch, 1, 1)
        self.cv2 = Conv(in_ch, hidden_ch, 1, 1)
        self.cv3 = Conv(2 * hidden_ch, out_ch, 1)
        self.m = nn.Sequential(*(Bottleneck(hidden_ch, hidden_ch, shortcut) for _ in range(n)))
    
    def forward(self, x):
        return self.cv3.forward(torch.cat([self.m.forward(self.cv1.forward(x)), self.cv2.forward(x)], dim=1))


class Bottleneck(nn.Module):
    """标准 Bottleneck"""
    def __init__(self, in_ch, out_ch, shortcut=True):
        super().__init__()
        self.cv1 = Conv(in_ch, out_ch, 3, 1)
        self.cv2 = Conv(out_ch, out_ch, 3, 1)
        self.add = shortcut and in_ch == out_ch
    
    def forward(self, x):
        return x + self.cv2.forward(self.cv1.forward(x)) if self.add else self.cv2.forward(self.cv1.forward(x))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF)"""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hidden_ch = in_ch // 2
        self.cv1 = Conv(in_ch, hidden_ch, 1, 1)
        self.cv2 = Conv(hidden_ch * 4, out_ch, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x):
        x = self.cv1.forward(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2.forward(torch.cat([x, y1, y2, self.m(y2)], 1))


class YOLOv5Seg(nn.Module):
    """YOLOv5-Seg 分割网络
    
    采用 YOLOv5 的 CSPDarknet 主干 + FPN + 分割头
    """
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        
        # Backbone (CSPDarknet)
        self.stem = Conv(in_ch, 64, 6, 2, 2)  # P1/2
        
        self.stage1 = nn.Sequential(
            Conv(64, 128, 3, 2),  # P2/4
            C3(128, 128, n=3)
        )
        
        self.stage2 = nn.Sequential(
            Conv(128, 256, 3, 2),  # P3/8
            C3(256, 256, n=6)
        )
        
        self.stage3 = nn.Sequential(
            Conv(256, 512, 3, 2),  # P4/16
            C3(512, 512, n=9)
        )
        
        self.stage4 = nn.Sequential(
            Conv(512, 1024, 3, 2),  # P5/32
            C3(1024, 1024, n=3),
            SPPF(1024, 1024, k=5)
        )
        
        # Neck (FPN + PAN)
        # Top-down
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_p4 = C3(1024 + 512, 512, n=3, shortcut=False)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_p3 = C3(512 + 256, 256, n=3, shortcut=False)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_p2 = C3(256 + 128, 128, n=3, shortcut=False)
        
        # Segmentation Head
        self.seg_head = nn.Sequential(
            Conv(128, 64, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            Conv(64, 32, 3, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, out_ch, 1)
        )
    
    def forward(self, x):
        # Backbone
        p1 = self.stem.forward(x)
        p2 = self.stage1(p1)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        
        # Neck - Top-down pathway
        x = self.up1(p5)
        x = torch.cat([x, p4], dim=1)
        x = self.c3_p4.forward(x)
        
        x = self.up2(x)
        x = torch.cat([x, p3], dim=1)
        x = self.c3_p3.forward(x)
        
        x = self.up3(x)
        x = torch.cat([x, p2], dim=1)
        x = self.c3_p2.forward(x)
        
        # Segmentation head
        out = self.seg_head(x)
        
        return out


class EAEFNet(nn.Module):
    """EAEFNet: Edge Attention Efficient Feature Network
    
    基于 FPN (Feature Pyramid Network) 架构，使用轻量级编码器。
    使用 segmentation_models_pytorch 库的 FPN 实现。
    
    特点:
    - 轻量级编码器 (MobileNetV2/EfficientNet)
    - 多尺度特征融合
    - 适合实时分割任务
    """
    def __init__(self, in_ch=1, out_ch=1, encoder_name='mobilenet_v2', encoder_weights=None):
        super().__init__()
        
        # 使用 segmentation_models_pytorch 的 FPN
        # FPN 擅长多尺度特征融合，适合边缘检测
        self.model = smp.FPN(
            encoder_name=encoder_name,  # 可选: mobilenet_v2, efficientnet-b0, resnet18 等
            encoder_weights=encoder_weights,  # None 或 'imagenet'
            in_channels=in_ch,
            classes=out_ch,
            activation=None  # 使用原始输出，由损失函数处理
        )
    
    def forward(self, x):
        return self.model(x)


class SegformerSMP(nn.Module):
    """基于 segmentation_models_pytorch 的 Segformer
    
    使用预训练的 MIT (Mix Transformer) 编码器，轻量高效。
    支持多种编码器规模：mit_b0 到 mit_b5
    """
    def __init__(self, in_ch=1, out_ch=1, encoder_name='mit_b0', encoder_weights=None):
        super().__init__()
        
        # 使用 segmentation_models_pytorch 的 Segformer
        self.model = smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,  # None 表示随机初始化，'imagenet' 表示使用预训练权重
            in_channels=in_ch,
            classes=out_ch
        )
    
    def forward(self, x):
        return self.model(x)


class PSPNet(nn.Module):
    """PSPNet: Pyramid Scene Parsing Network
    
    使用 segmentation_models_pytorch 库实现。
    特点：
    - 金字塔池化模块（PPM）聚合多尺度上下文
    - 适合高分辨率图像分割
    - 支持多种编码器（ResNet、MobileNet等）
    """
    def __init__(self, in_ch=1, out_ch=1, encoder_name='resnet34', encoder_weights=None):
        super().__init__()
        
        # 使用 segmentation_models_pytorch 的 PSPNet
        self.model = smp.PSPNet(
            encoder_name=encoder_name,  # 可选: resnet18/34/50, mobilenet_v2 等
            encoder_weights=encoder_weights,  # None 或 'imagenet'
            in_channels=in_ch,
            classes=out_ch,
            psp_out_channels=512,  # PSP 模块输出通道数
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            activation=None
        )
    
    def forward(self, x):
        return self.model(x)


class MFNet(nn.Module):
    """MFNet: Multi-spectral Fusion Network
    
    轻量级多光谱图像分割网络，参数量约为 SegNet 的 1/40。
    原始设计用于 RGB-Thermal 融合，这里适配为单通道版本。
    
    架构：编码器-解码器结构，使用残差连接和特征融合
    """
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        
        # 编码器
        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(32, out_ch, 1)
    
    def forward(self, x):
        # 编码
        e1 = self.encoder_conv1(x)
        e1_pool = self.pool1(e1)
        
        e2 = self.encoder_conv2(e1_pool)
        e2_pool = self.pool2(e2)
        
        e3 = self.encoder_conv3(e2_pool)
        e3_pool = self.pool3(e3)
        
        # 瓶颈
        b = self.bottleneck(e3_pool)
        
        # 解码
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder_conv3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder_conv2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder_conv1(d1)
        
        out = self.final_conv(d1)
        return out


class RTFNet(nn.Module):
    """RTFNet: RGB-Thermal Fusion Network
    
    用于城市场景语义分割的 RGB-热红外融合网络。
    原始设计用于双模态融合，这里适配为单通道版本。
    
    特点：
    - 双编码器结构（这里简化为单编码器）
    - 多层次特征融合
    - 残差连接
    """
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        
        # 编码器（简化版，单通道）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.encoder4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # 瓶颈
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(64, out_ch, 1)
    
    def forward(self, x):
        # 编码
        e1 = self.encoder1(x)
        e1_pool = self.pool1(e1)
        
        e2 = self.encoder2(e1_pool)
        e2_pool = self.pool2(e2)
        
        e3 = self.encoder3(e2_pool)
        e3_pool = self.pool3(e3)
        
        e4 = self.encoder4(e3_pool)
        e4_pool = self.pool4(e4)
        
        # 瓶颈
        b = self.bottleneck(e4_pool)
        
        # 解码
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        out = self.final_conv(d1)
        return out


class FEANet(nn.Module):
    """FEANet: Feature-Enhanced Attention Network
    
    IROS 2021 论文提出的 RGB-Thermal 分割网络。
    原始设计用于双模态融合，这里适配为单通道版本。
    
    特点：
    - 特征增强注意力机制
    - 轻量级设计
    - 适合实时应用
    """
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        
        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # 编码器块
        self.enc1 = self._make_encoder_block(32, 64)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)
        
        # 注意力模块
        self.att1 = ChannelAttentionModule(64, reduction=4)
        self.att2 = ChannelAttentionModule(128, reduction=8)
        self.att3 = ChannelAttentionModule(256, reduction=16)
        self.att4 = ChannelAttentionModule(512, reduction=32)
        
        # 解码器
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._make_decoder_block(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._make_decoder_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._make_decoder_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = self._make_decoder_block(64, 32)
        
        self.final_conv = nn.Conv2d(32, out_ch, 1)
    
    def _make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 初始特征
        x0 = self.conv1(x)
        
        # 编码 + 注意力
        e1 = self.enc1(x0)
        e1 = self.att1.forward(e1)
        
        e2 = self.enc2(e1)
        e2 = self.att2.forward(e2)
        
        e3 = self.enc3(e2)
        e3 = self.att3.forward(e3)
        
        e4 = self.enc4(e3)
        e4 = self.att4.forward(e4)
        
        # 解码
        d4 = self.up4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return out


# 模型工厂：注册所有可用模型
MODEL_FACTORY: Dict[str, Callable[..., nn.Module]] = {
    "unet": UNet,
    "unet_gcm": UNetGCMOld,
    "yolov5seg": YOLOv5Seg,
    "eaefnet": EAEFNet,  # 基于 FPN 的轻量级网络
    "segformer": SegformerSMP,  # 使用 SMP 库的 Segformer
    "pspnet": PSPNet,  # 金字塔场景解析网络
    "mfnet": MFNet,  # 多光谱融合网络（轻量级）
    "rtfnet": RTFNet,  # RGB-Thermal 融合网络
    "feanet": FEANet  # 特征增强注意力网络
}


def build_model(name: str, in_ch: int, out_ch: int) -> nn.Module:
    """根据名称构建模型
    
    Args:
        name: 模型名称，必须在 MODEL_FACTORY 中注册
        in_ch: 输入通道数
        out_ch: 输出通道数
    
    Returns:
        构建好的模型实例
    
    Raises:
        ValueError: 如果模型名称不存在
    """
    try:
        return MODEL_FACTORY[name](in_ch=in_ch, out_ch=out_ch)
    except KeyError as exc:
        raise ValueError(f"Unsupported model '{name}'. Available models: {list(MODEL_FACTORY.keys())}") from exc
