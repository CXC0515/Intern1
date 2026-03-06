import os
import cv2
import glob
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# --- 路径配置 ---
IMAGE_DIR = r"E:\code\air_mask\images"
OUTPUT_ROOT = r"E:\code\air_mask\results_compare"  # 结果保存的总目录

# --- 模型 A (新模型) ---
MODEL_A_PATH = r"E:\code\air_mask\person.pt"
MODEL_A_NAME = "New_Model"
# 新模型中 'air_mask' 的 Class ID (根据你之前的描述是 3)
ID_A = 3 

# --- 模型 B (旧模型) ---
MODEL_B_PATH = r"E:\code\air_mask\air_mask_m_640.pt"
MODEL_B_NAME = "Old_Model"
# 旧模型中 'air_mask' 的 Class ID
# ⚠️ 注意：如果旧模型是专门只训练了空气呼吸器单类，ID 通常是 0
# 如果旧模型也是多类，请修改这里。这里默认设为 0。
ID_B = 0 

# 公共配置
CONF_THRESHOLD = 0.15

# ================= 🚀 主程序 =================

def run_inference(model, image_path, target_id):
    """
    辅助函数：跑单个模型的推理，返回 (数量, 最高置信度, 绘图结果)
    """
    results = model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)[0]
    
    count = 0
    max_conf = 0.0
    
    # 统计目标
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        if cls_id == target_id:
            count += 1
            if conf > max_conf:
                max_conf = conf
    
    # 绘图
    plotted_img = results.plot(line_width=2)
    return count, max_conf, plotted_img

def main():
    # 1. 准备目录
    dir_a = os.path.join(OUTPUT_ROOT, MODEL_A_NAME)
    dir_b = os.path.join(OUTPUT_ROOT, MODEL_B_NAME)
    
    for d in [OUTPUT_ROOT, dir_a, dir_b]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 2. 加载模型
    print(f"[-] 正在加载模型 A: {MODEL_A_PATH}")
    try:
        model_a = YOLO(MODEL_A_PATH)
    except Exception as e:
        print(f"[错误] 模型 A 加载失败: {e}")
        return

    print(f"[-] 正在加载模型 B: {MODEL_B_PATH}")
    try:
        model_b = YOLO(MODEL_B_PATH)
    except Exception as e:
        print(f"[错误] 模型 B 加载失败: {e}")
        return

    # 3. 扫描图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_files:
        print(f"[警告] {IMAGE_DIR} 下没有图片！")
        return

    print(f"[-] 开始对比测试，共 {len(image_files)} 张图片...")

    report_data = []

    # 4. 循环对比
    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        
        # --- 跑模型 A ---
        count_a, conf_a, img_a = run_inference(model_a, img_path, ID_A)
        # 保存 A 的图
        cv2.imwrite(os.path.join(dir_a, filename), img_a)
        
        # --- 跑模型 B ---
        count_b, conf_b, img_b = run_inference(model_b, img_path, ID_B)
        # 保存 B 的图
        cv2.imwrite(os.path.join(dir_b, filename), img_b)

        # --- 记录数据 ---
        # 判定胜负/差异
        diff_msg = "一致"
        if count_a > 0 and count_b == 0:
            diff_msg = "🔥新模型检出(旧模型漏检)"
        elif count_a == 0 and count_b > 0:
            diff_msg = "⚠️旧模型检出(新模型漏检)"
        elif count_a > count_b:
            diff_msg = "新模型数量更多"
        elif count_b > count_a:
            diff_msg = "旧模型数量更多"
            
        report_data.append({
            "文件名": filename,
            f"{MODEL_A_NAME}_数量": count_a,
            f"{MODEL_A_NAME}_置信度": round(conf_a, 4),
            f"{MODEL_B_NAME}_数量": count_b,
            f"{MODEL_B_NAME}_置信度": round(conf_b, 4),
            "对比结论": diff_msg
        })

    # 5. 生成对比报表
    df = pd.DataFrame(report_data)
    
    # 排序优化：把有差异的排在最前面，方便 PM 检查
    # 逻辑：如果 '对比结论' 不是 '一致'，排在前面
    df['is_diff'] = df['对比结论'].apply(lambda x: 0 if x == '一致' else 1)
    df = df.sort_values(by=['is_diff', '文件名'], ascending=[False, True])
    df = df.drop(columns=['is_diff']) # 导出前删掉辅助列

    csv_path = os.path.join(OUTPUT_ROOT, "Model_Comparison_Report.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("【PK 结束】")
    print(f"对比报告: {csv_path}")
    print(f"新模型结果图: {dir_a}")
    print(f"旧模型结果图: {dir_b}")
    print("提示: 请打开 CSV，筛选 '对比结论' 列，重点查看不一致的样本。")
    print("="*50)

if __name__ == "__main__":
    main()