import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from train import build_model

# ========== 配置区域 ==========
CHECKPOINT_PATH = "runs/unetmodel/best_model.pth"  # 模型权重路径
MODEL_NAME = "unet"  # 模型名称: unet 或 slimfaster_unet
IMAGE_DIR = "Gas_DB/T"  # 图像目录
LABEL_DIR = "Gas_DB/visual_labels"  # 标签目录
IMAGE_ID = "00001"  # 要可视化的图像ID
DEVICE = "cuda"  # cuda 或 cpu
THRESHOLD = 0.5  # 二值化阈值
# ==============================

def main():
    # 设备检查
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = build_model(MODEL_NAME, in_ch=1, out_ch=1, n_div=4)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载图像
    image_path = f"{IMAGE_DIR}/{IMAGE_ID}.png"
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("L")
    img_tensor = transform(img)
    img_np = np.array(img.resize((640, 512)))
    
    # 加载标签
    label_path = f"{LABEL_DIR}/{IMAGE_ID}.png"
    print(f"Loading label: {label_path}")
    label = Image.open(label_path).convert("L")
    label_np = np.array(label.resize((640, 512)))
    label_np = (label_np > 127).astype(np.float32)
    
    # 推理
    print("Running inference...")
    with torch.no_grad():
        img_input = img_tensor.unsqueeze(0).to(device)
        logits = model(img_input)
        probs = torch.sigmoid(logits)
        pred = (probs > THRESHOLD).float()
    
    pred_np = pred.squeeze().cpu().numpy()
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(label_np, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    # 保存可视化结果
    output_path = f"visualizations/{MODEL_NAME}_{IMAGE_ID}.png"
    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()
    
    print("Done!")


if __name__ == "__main__":
    main()
