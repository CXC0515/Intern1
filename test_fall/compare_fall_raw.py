import os
import cv2
import glob
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# --- 路径配置 ---
# 待测试图片目录
IMAGE_DIR = r"E:\code\test_fall\跌落检测蹲姿测试\test_images"
# 结果保存目录
OUTPUT_ROOT = r"E:\code\test_fall\results_compare_raw"

# --- 模型 A (你的多模态模型) ---
MODEL_A_PATH = r"E:\code\test_fall\person.pt" 
MODEL_A_NAME = "Multimodal"
# 多模态中 'fall' 的 Class ID (根据你之前的映射表，摔倒的ID是 7)
# ⚠️ 如果你的 person.pt 里的 ID 变了，请在这里修改
ID_A = 0 

# --- 模型 B (旧 - 单模态 shuailuo.pt) ---
MODEL_B_PATH = r"E:\code\test_fall\models\shuailuo.pt"
MODEL_B_NAME = "Single_Shuailuo"
# 单模态中 'fall' 的 Class ID (通常是 0)
ID_B = 0 

# --- 唯一裁判：置信度 ---
# 只要高于这个值，就算摔倒
CONF_THRESHOLD = 0.1 

# ================= 🚀 主程序 =================

def run_raw_inference(model, image_path, target_id):
    """
    纯净推理函数：不做任何宽高比过滤
    返回: (数量, 最高置信度, 绘图结果)
    """
    try:
        # 1. 推理
        results = model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)[0]
        
        count = 0
        max_conf = 0.0
        
        # 复制图片用于画图
        plotted_img = results.orig_img.copy()
        
        # 2. 遍历检测框
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 只要类别对，就计入
            if cls_id == target_id:
                count += 1
                if conf > max_conf:
                    max_conf = conf
                
                # 简单的画框逻辑 (只画红框)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = f"FALL {conf:.2f}"
                
                # 画框 - 红色
                cv2.rectangle(plotted_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # 写字
                cv2.putText(plotted_img, label, (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            
        return count, max_conf, plotted_img

    except Exception as e:
        print(f"[警告] 推理出错: {image_path} | {e}")
        # 出错时返回原图
        return 0, 0.0, cv2.imread(image_path)

def main():
    # 1. 准备目录
    dir_a = os.path.join(OUTPUT_ROOT, MODEL_A_NAME)
    dir_b = os.path.join(OUTPUT_ROOT, MODEL_B_NAME)
    
    for d in [OUTPUT_ROOT, dir_a, dir_b]:
        if not os.path.exists(d):
            os.makedirs(d)

    # 2. 加载模型
    print(f"[-] 正在加载 A (多模态): {MODEL_A_PATH}")
    try:
        model_a = YOLO(MODEL_A_PATH)
    except Exception as e:
        print(f"[错误] 模型 A 加载失败: {e}")
        return

    print(f"[-] 正在加载 B (单模态): {MODEL_B_PATH}")
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

    print(f"[-] 开始纯净对比测试 (Raw Mode)，共 {len(image_files)} 张图片...")

    report_data = []

    # 4. 循环对比
    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        
        # --- 跑模型 A ---
        count_a, conf_a, img_a = run_raw_inference(model_a, img_path, ID_A)
        cv2.imwrite(os.path.join(dir_a, filename), img_a)
        
        # --- 跑模型 B ---
        count_b, conf_b, img_b = run_raw_inference(model_b, img_path, ID_B)
        cv2.imwrite(os.path.join(dir_b, filename), img_b)

        # --- 记录数据 ---
        # 这里的逻辑是：谁检出了，谁就赢（或者谁检出的更多）
        diff_msg = "一致"
        if count_a > 0 and count_b == 0:
            diff_msg = "🔥多模态检出(单模漏)" # 重点关注：多模态是否更强？
        elif count_a == 0 and count_b > 0:
            diff_msg = "⚠️单模态检出(多模漏)" # 重点关注：多模态是否变瞎了？
        elif count_a > count_b:
            diff_msg = "多模态数量更多"
        elif count_b > count_a:
            diff_msg = "单模态数量更多"
            
        report_data.append({
            "文件名": filename,
            "对比结论": diff_msg,
            f"{MODEL_A_NAME}_数量": count_a,
            f"{MODEL_A_NAME}_Conf": round(conf_a, 4),
            f"{MODEL_B_NAME}_数量": count_b,
            f"{MODEL_B_NAME}_Conf": round(conf_b, 4),
        })

    # 5. 生成报告
    df = pd.DataFrame(report_data)
    
    # 排序优化：把有差异的排在最前面
    df['is_diff'] = df['对比结论'].apply(lambda x: 0 if x == '一致' else 1)
    df = df.sort_values(by=['is_diff', '文件名'], ascending=[False, True])
    df = df.drop(columns=['is_diff']) 

    csv_path = os.path.join(OUTPUT_ROOT, "Fall_Model_Raw_Comparison.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("【纯净对比结束】")
    print(f"📊 CSV报告: {csv_path}")
    print(f"📂 图片保存至: {OUTPUT_ROOT}")
    print("👉 请打开 CSV 筛选 '⚠️单模态检出(多模漏)' 和 '🔥多模态检出(单模漏)' 查看差异样本")
    print("="*50)

if __name__ == "__main__":
    main()