import os
import cv2
import glob
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm  

# ================= 配置区域 =================
# 模型路径
MODEL_PATH = r"E:\code\air_mask\person.pt"

# 待测试图片文件夹
IMAGE_DIR = r"E:\code\air_mask\images"

# 结果保存文件夹
OUTPUT_DIR = r"E:\code\air_mask\results"

# 你的类别映射表中，空气呼吸器(air_mask)的 ID 是多少？
# 根据你之前的信息: {3: 'air_mask'}
TARGET_CLASS_ID = 3  
TARGET_CLASS_NAME = "air_mask"

# 置信度阈值 (低于这个不画框/不统计)
CONF_THRESHOLD = 0.15 

# ================= 主程序 =================

def main():
    # 1. 准备目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[-] 创建输出目录: {OUTPUT_DIR}")
    else:
        print(f"[-] 输出目录已存在: {OUTPUT_DIR}")

    # 2. 加载模型
    print(f"[-] 正在加载模型: {MODEL_PATH} ...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return

    # 3. 获取所有图片
    # 支持 jpg, png, jpeg, bmp
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        # recursive=False 表示只看当前目录，不查子目录
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_files:
        print(f"[警告] 在 {IMAGE_DIR} 下没有找到图片！")
        return

    print(f"[-] 找到 {len(image_files)} 张图片，开始推理...")

    # 用于存放统计数据
    report_data = []

    # 4. 循环推理 (使用 tqdm 显示进度条)
    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        
        # 推理
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        
        # 统计该图中的空气呼吸器数量
        target_count = 0
        max_conf = 0.0
        
        # 遍历检测框
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id == TARGET_CLASS_ID:
                target_count += 1
                if conf > max_conf:
                    max_conf = conf

        # 绘制结果图
        plotted_img = results.plot(line_width=2, font_size=2)
        
        # 如果需要在图上醒目提示是否检测到 (可选)
        # if target_count == 0:
        #     cv2.putText(plotted_img, "MISSING MASK", (10, 50), 
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # 保存图片
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, plotted_img)

        # 记录数据
        report_data.append({
            "文件名": filename,
            f"{TARGET_CLASS_NAME}_数量": target_count,
            "最高置信度": round(max_conf, 4),
            "检测结果": "✅有" if target_count > 0 else "❌无",
            "保存路径": save_path
        })

    # 5. 生成统计报表
    csv_path = os.path.join(OUTPUT_DIR, "detection_report.csv")
    df = pd.DataFrame(report_data)
    
    # 将“无”的排在前面，方便PM优先检查漏检
    df = df.sort_values(by=f"{TARGET_CLASS_NAME}_数量", ascending=True)
    
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("【测试完成】")
    print(f"处理图片: {len(image_files)} 张")
    print(f"结果图片位置: {OUTPUT_DIR}")
    print(f"统计报表位置: {csv_path}")
    print("提示: 请打开 report.csv，筛选 '❌无' 来检查漏检样本。")
    print("="*50)

if __name__ == "__main__":
    main()