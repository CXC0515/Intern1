from ultralytics import YOLO

# 模型路径
model_path = r"E:\code\model_test\fall2_m_640.pt"

try:
    model = YOLO(model_path)
    print("\n" + "="*30)
    print("模型类别映射表 (Copy keys to LABEL_MAP):")
    print("="*30)
    # 打印 ID 和 名称
    print(model.names)
    print("="*30 + "\n")
    
    # 如果你想看更清晰的列表格式：
    for class_id, class_name in model.names.items():
        print(f"ID: {class_id}  ->  名称: {class_name}")

except Exception as e:
    print(f"加载失败: {e}")