"""
损失函数和评估指标模块
"""
import torch
import torch.nn as nn


def dice_coefficient(logits, target, smooth=1e-5):
    """计算 Dice 系数（用于评估）
    
    Args:
        logits: 模型输出的 logits，未经 sigmoid
        target: 真实标签 (0/1)
        smooth: 平滑项，避免除零
    
    Returns:
        Dice 系数 [0, 1]，越大越好
    """
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class FocalLoss(nn.Module):
    """Focal Loss 用于解决类别不平衡问题
    
    对难分类样本给予更高权重，减少易分类样本的损失贡献。
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡因子，用于平衡正负样本
            gamma: 聚焦参数，gamma越大对难分类样本的关注度越高
            reduction: 损失聚合方式
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: 模型输出 logits [B, 1, H, W]
            target: 真实标签 [B, 1, H, W]
        
        Returns:
            Focal Loss 值
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)  # pt = p if target==1 else (1-p)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceFocalLoss(nn.Module):
    """Dice Loss + Focal Loss 组合损失函数
    
    结合区域重叠度和难分类样本聚焦，适用于分割任务中的类别不平衡问题。
    """
    def __init__(self, smooth=1e-5, alpha=1.0, gamma=2.0):
        """
        Args:
            smooth: Dice Loss 的平滑项
            alpha: Focal Loss 的平衡因子
            gamma: Focal Loss 的聚焦参数
        """
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: 模型输出 logits [B, 1, H, W]
            target: 真实标签 [B, 1, H, W]
        
        Returns:
            组合损失值
        """
        focal_loss = self.focal_loss(pred, target)
        
        # Dice Loss
        probs = torch.sigmoid(pred)
        pred_flat = probs.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 0.5 * focal_loss + 0.5 * dice_loss
