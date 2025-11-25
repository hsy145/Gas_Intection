def print_results(model_name, acc, recall, iou, dice):
    """打印预期结果

    Args:
        model_name: 模型名称
    """
    print(f"Recall: {recall:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")