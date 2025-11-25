"""
多模型对比可视化脚本
根据 checkpoints 生成类似论文 Figure 7 的对比结果
"""
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

from models import build_model
from dataset import load_ids


def load_model_checkpoint(model_name, checkpoint_path, device):
    """加载模型和权重
    
    Args:
        model_name: 模型名称
        checkpoint_path: checkpoint 路径
        device: 设备
    
    Returns:
        加载好的模型
    """
    model = build_model(model_name, in_ch=1, out_ch=1)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_single_image(model, img_path, transform, device, threshold=0.5):
    """对单张图像进行预测
    
    Args:
        model: 模型
        img_path: 图像路径
        transform: 变换
        device: 设备
        threshold: 二值化阈值
    
    Returns:
        预测结果 numpy 数组
    """
    img = Image.open(img_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()
    
    return pred.squeeze().cpu().numpy()


def visualize_comparison(image_ids, models_dict, data_root, output_dir, device, img_size=(512, 640)):
    """生成多模型对比可视化
    
    Args:
        image_ids: 要可视化的图像 ID 列表
        models_dict: 模型字典 {model_name: checkpoint_path}
        data_root: 数据集根目录
        output_dir: 输出目录
        device: 设备
        img_size: 图像尺寸 (H, W)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载所有模型
    print("Loading models...")
    loaded_models = {}
    for model_name, ckpt_path in models_dict.items():
        if os.path.exists(ckpt_path):
            print(f"  - {model_name}: {ckpt_path}")
            loaded_models[model_name] = load_model_checkpoint(model_name, ckpt_path, device)
        else:
            print(f"  - {model_name}: checkpoint not found, skipping")
    
    if not loaded_models:
        print("No valid models found!")
        return
    
    # 对每张图像生成对比
    print(f"\nGenerating visualizations for {len(image_ids)} images...")
    for img_id in tqdm(image_ids):
        # 加载原始图像和标签
        img_path = f"{data_root}/T/{img_id}.png"
        label_path = f"{data_root}/visual_labels/{img_id}.png"
        
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            print(f"Skipping {img_id}: file not found")
            continue
        
        # 热图
        thermal_img = np.array(Image.open(img_path).convert("L").resize((img_size[1], img_size[0])))
        
        # Ground Truth
        gt_img = np.array(Image.open(label_path).convert("L").resize((img_size[1], img_size[0])))
        gt_mask = (gt_img > 127).astype(np.uint8) * 255
        
        # 预测结果
        predictions = {}
        for model_name, model in loaded_models.items():
            pred = predict_single_image(model, img_path, transform, device)
            predictions[model_name] = (pred * 255).astype(np.uint8)
        
        # 创建对比图
        num_rows = 2 + len(loaded_models)  # Thermal + GT + 各模型
        fig, axes = plt.subplots(num_rows, 1, figsize=(8, num_rows * 1.5))
        
        # Thermal
        axes[0].imshow(thermal_img, cmap='gray')
        axes[0].set_title('Thermal', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(gt_mask, cmap='gray')
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # 各模型预测
        for idx, (model_name, pred_mask) in enumerate(predictions.items()):
            axes[2 + idx].imshow(pred_mask, cmap='gray')
            axes[2 + idx].set_title(model_name.upper(), fontsize=12, fontweight='bold')
            axes[2 + idx].axis('off')
        
        plt.tight_layout()
        save_path = output_dir / f"comparison_{img_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"\nVisualization complete! Results saved to: {output_dir}")


def visualize_grid_comparison(image_ids, models_dict, data_root, output_path, device, img_size=(512, 640)):
    """生成网格对比可视化（类似论文 Figure 7）
    
    Args:
        image_ids: 要可视化的图像 ID 列表
        models_dict: 模型字典 {model_name: checkpoint_path}
        data_root: 数据集根目录
        output_path: 输出文件路径
        device: 设备
        img_size: 图像尺寸 (H, W)
    """
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载所有模型
    print("Loading models...")
    loaded_models = {}
    model_order = []
    
    # 定义模型显示名称映射
    model_display_names = {
        'unet': 'UNet',
        'yolov5seg': 'YOLOv5-Seg',
        'eaefnet': 'EAEFNet',
        'segformer': 'SegFormer',
        'pspnet': 'PSPNet',
        'mfnet': 'MFNet',
        'rtfnet': 'RTFNet',
        'feanet': 'FEANet',
        'unet_gcm': 'Ours'
    }
    
    for model_name, ckpt_path in models_dict.items():
        if os.path.exists(ckpt_path):
            print(f"  - {model_name}: {ckpt_path}")
            loaded_models[model_name] = load_model_checkpoint(model_name, ckpt_path, device)
            model_order.append(model_name)
        else:
            print(f"  - {model_name}: checkpoint not found, skipping")

    if 'unet_gcm' in model_order:
        model_order.remove('unet_gcm')
        model_order.append('unet_gcm')
    
    if not loaded_models:
        print("No valid models found!")
        return
    
    # 收集所有图像数据
    print(f"\nProcessing {len(image_ids)} images...")
    all_data = []
    
    for img_id in tqdm(image_ids):
        img_path = f"{data_root}/T/{img_id}.png"
        label_path = f"{data_root}/visual_labels/{img_id}.png"
        
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue
        
        # 热图
        thermal_img = np.array(Image.open(img_path).convert("L").resize((img_size[1], img_size[0])))
        
        # Ground Truth
        gt_img = np.array(Image.open(label_path).convert("L").resize((img_size[1], img_size[0])))
        gt_mask = (gt_img > 127).astype(np.uint8) * 255
        
        # 预测结果
        predictions = {}
        for model_name in model_order:
            model = loaded_models[model_name]
            pred = predict_single_image(model, img_path, transform, device)
            # 将预测结果转为紫色可视化
            pred_colored = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
            pred_colored[pred > 0.5] = [128, 0, 255]  # 紫色
            predictions[model_name] = pred_colored
        
        all_data.append({
            'thermal': thermal_img,
            'gt': gt_mask,
            'predictions': predictions
        })
    
    # 创建网格可视化
    num_images = len(all_data)
    num_rows = 2 + len(model_order)  # Thermal + GT + 各模型
    
    fig, axes = plt.subplots(num_rows, num_images, figsize=(num_images * 2, num_rows * 2))
    
    if num_images == 1:
        axes = axes.reshape(-1, 1)
    
    row_labels = ['Thermal', 'Ground\nTruth'] + [model_display_names.get(m, m.upper()) for m in model_order]
    
    for col_idx, data in enumerate(all_data):
        # Thermal
        axes[0, col_idx].imshow(data['thermal'], cmap='gray')
        axes[0, col_idx].axis('off')
        
        # Ground Truth（紫色可视化）
        gt_colored = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        gt_colored[data['gt'] > 127] = [128, 0, 255]
        axes[1, col_idx].imshow(gt_colored)
        axes[1, col_idx].axis('off')
        
        # 各模型预测
        for model_idx, model_name in enumerate(model_order):
            axes[2 + model_idx, col_idx].imshow(data['predictions'][model_name])
            axes[2 + model_idx, col_idx].axis('off')
    
    # 在第一列左侧添加文本标签
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(-0.1, 0.5, label, transform=axes[row_idx, 0].transAxes,
                              fontsize=14, fontweight='bold', ha='right', va='center')
    
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.15)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nGrid visualization saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model comparison visualization")
    parser.add_argument('--root', default='Gas_DB', help='Dataset root directory')
    parser.add_argument('--test-file', default='Gas_DB/test.txt', help='Test IDs file')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--output-dir', default='visualizations/comparison/test', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=8, help='Number of samples to visualize')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--grid', action='store_true', help='Generate grid layout')
    parser.add_argument('--img-size', nargs=2, type=int, default=[512, 640], metavar=('H', 'W'), help='Image size')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 加载测试集 ID
    with open(args.test_file, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]
    
    # 选择要可视化的样本
    if args.num_samples == 8:
        selected_ids = [
            test_ids[0],   # 第1列: 00131
            test_ids[9],   # 第2列: 00565
            test_ids[11],  # 第3列: 00664
            test_ids[14],  # 第4列: 00509
            test_ids[4],   # 第5列: 01218
            test_ids[5],   # 第6列: 00126
            test_ids[6],   # 第7列: 00335
            test_ids[7],   # 第8列: 00462
        ]
    else:
        selected_ids = test_ids[:args.num_samples]
    print(f"Selected {len(selected_ids)} samples for visualization")
    
    # 定义模型和对应的 checkpoint
    models_dict = {
        'unet': f'{args.checkpoint_dir}/unetmodel/best_model.pth',
        'yolov5seg': f'{args.checkpoint_dir}/yolov5segmodel/best_model.pth',
        'eaefnet': f'{args.checkpoint_dir}/eaefnetmodel/best_model.pth',
        'segformer': f'{args.checkpoint_dir}/segformermodel/best_model.pth',
        'pspnet': f'{args.checkpoint_dir}/pspnetmodel/best_model.pth',
        'mfnet': f'{args.checkpoint_dir}/mfnetmodel/best_model.pth',
        'rtfnet': f'{args.checkpoint_dir}/rtfnetmodel/best_model.pth',
        'feanet': f'{args.checkpoint_dir}/feanetmodel/best_model.pth',
        'unet_gcm': f'{args.checkpoint_dir}/unet_gcmmodel/best_model.pth',
    }
    
    if args.grid:
        # 生成网格布局（类似论文）
        output_path = f"{args.output_dir}/grid_comparison.png"
        visualize_grid_comparison(selected_ids, models_dict, args.root, output_path, device, tuple(args.img_size))
    else:
        # 生成单独的对比图
        visualize_comparison(selected_ids, models_dict, args.root, args.output_dir, device, tuple(args.img_size))


if __name__ == "__main__":
    main()
