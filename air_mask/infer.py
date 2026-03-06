import os
from ultralytics import YOLO

def main():
    # 1. 路径设置 (使用相对路径，确保在任何电脑上都能运行)
    # 获取当前脚本(infer.py)所在的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型路径
    model_path = os.path.join(base_dir, 'air_mask_m_640.pt')
    # 待检测图片文件夹路径
    source_path = os.path.join(base_dir, 'images')
    # 结果保存路径
    save_dir = os.path.join(base_dir, 'results')

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 -> {model_path}")
        return

    # 检查图片文件夹是否存在
    if not os.path.exists(source_path):
        print(f"错误：找不到图片文件夹 -> {source_path}")
        return

    print("正在加载模型...")
    
    try:
        # 2. 加载模型
        model = YOLO(model_path)
        
        # 3. 执行推理
        # source: 图片源
        # save: 是否保存结果图片
        # project: 结果保存的根目录
        # name: 结果保存的子目录 (设为 . 代表直接保存在 project 下)
        # conf: 置信度阈值 (0.25 是默认值，可根据需要调整)
        print(f"开始检测 {source_path} 中的图片...")
        
        results = model.predict(
            source=source_path, 
            save=True, 
            project=save_dir, 
            name='output',  # 结果会保存在 results/output 文件夹下
            exist_ok=True,  # 如果文件夹已存在，不报错
            conf=0.25       # 置信度，低于这个分数的框不会显示
        )

        print(f"\n检测完成！请查看结果文件夹: {os.path.join(save_dir, 'output')}")

    except Exception as e:
        print(f"发生错误: {e}")
        print("提示：如果报错模型格式不对，请确认该 .pt 文件是使用 Ultralytics 框架训练的。")

if __name__ == "__main__":
    main()