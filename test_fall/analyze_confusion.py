import os
import cv2
import glob
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# ================= ⚙️ 配置区域 =================

# --- 路径配置 ---
IMAGE_DIR = r"E:\code\test_fall\跌落检测蹲姿测试\test_images"
OUTPUT_ROOT = r"E:\code\test_fall\results_confusion_check"

# --- 模型 A: 多模态模型 ---
MODEL_MM_PATH = r"E:\code\test_fall\fall2_m_640.pt"
ID_MM_PERSON = 1  # Person
ID_MM_FALL = 0    # Fall

# --- 模型 B: 单模态跌倒 ---
MODEL_FALL_PATH = r"E:\code\test_fall\models\shuailuo.pt"
ID_SINGLE_FALL = 0

# --- 模型 C: 单模态行人 ---
MODEL_PERSON_PATH = r"E:\code\test_fall\models\xingren.pt"
ID_SINGLE_PERSON = 0

# 置信度阈值
CONF_THRESHOLD = 0.01

# --- 🎨 显眼包绘图参数 ---
FONT_SCALE = 1.2       # 字体放大 (1.2倍)
FONT_THICKNESS = 2     # 文字加粗
BOX_THICKNESS = 3      # 边框加粗

# ================= 🚀 核心逻辑 =================

def get_max_conf(model, img_path, target_id):
    """
    运行推理，返回 (数量, 最高置信度, 检测框列表, 原图)
    """
    try:
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        count = 0
        max_conf = 0.0
        boxes = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id == target_id:
                count += 1
                if conf > max_conf:
                    max_conf = conf
                boxes.append({
                    'xyxy': box.xyxy[0].tolist(),
                    'conf': conf
                })
        
        return count, max_conf, boxes, results.orig_img
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0.0, [], None

def draw_boxes(image, boxes, label_prefix, color):
    """
    画框+写大字，确保置信度一眼可见
    """
    for item in boxes:
        box = item['xyxy']
        conf = item['conf']
        x1, y1, x2, y2 = map(int, box)
        
        # 1. 画加粗边框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        
        # 2. 构造标签： "Fall: 0.85"
        label = f"{label_prefix}: {conf:.2f}"
        
        # 3. 计算文字背景大小
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        
        # 4. 画文字背景条 (填充颜色，让字在任何背景下都清楚)
        # 放在框的上方，如果不小心出界了就移到框内部
        text_y_bottom = y1
        text_y_top = y1 - h - 10
        
        if text_y_top < 0: # 如果上方没位置了，就画在框里面
            text_y_bottom = y1 + h + 10
            text_y_top = y1
            
        cv2.rectangle(image, (x1, text_y_top), (x1 + w, text_y_bottom), color, -1)
        
        # 5. 写白字
        cv2.putText(image, label, (x1, text_y_bottom - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
    return image

def main():
    # 1. 准备目录
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    
    dir_mm = os.path.join(OUTPUT_ROOT, "Multimodal_View")
    dir_single = os.path.join(OUTPUT_ROOT, "Single_Models_View")
    os.makedirs(dir_mm, exist_ok=True)
    os.makedirs(dir_single, exist_ok=True)

    # 2. 加载模型
    print(f"[-] 加载多模态模型: {MODEL_MM_PATH}")
    model_mm = YOLO(MODEL_MM_PATH)
    print(f"[-] 加载单模态跌倒: {MODEL_FALL_PATH}")
    model_fall = YOLO(MODEL_FALL_PATH)
    print(f"[-] 加载单模态行人: {MODEL_PERSON_PATH}")
    model_person = YOLO(MODEL_PERSON_PATH)

    # 3. 扫描图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    print(f"[-] 开始分析 (大字体版)...")

    report_data = []

    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        
        # --- A. 推理 ---
        mm_fall_cnt, mm_fall_conf, mm_fall_boxes, img_raw = get_max_conf(model_mm, img_path, ID_MM_FALL)
        mm_person_cnt, mm_person_conf, mm_person_boxes, _ = get_max_conf(model_mm, img_path, ID_MM_PERSON)
        
        s_fall_cnt, s_fall_conf, s_fall_boxes, _ = get_max_conf(model_fall, img_path, ID_SINGLE_FALL)
        s_person_cnt, s_person_conf, s_person_boxes, _ = get_max_conf(model_person, img_path, ID_SINGLE_PERSON)

        # --- B. 诊断 ---
        diagnosis = "其他"
        if mm_fall_cnt > 0:
            diagnosis = "✅多模态正确检出(Fall)"
        elif s_fall_cnt > 0 and mm_fall_cnt == 0:
            if mm_person_cnt > 0:
                diagnosis = "❌误识别为Person (实锤)"
            else:
                diagnosis = "⚠️多模态漏检 (无识别)"
        elif mm_person_cnt > 0 and s_fall_cnt == 0:
             diagnosis = "普通行人 (Person)"

        # --- C. 画图 1: 多模态视图 ---
        if img_raw is not None:
            img_mm_plot = img_raw.copy()
            # 蓝色画 Person
            draw_boxes(img_mm_plot, mm_person_boxes, "Person", (255, 0, 0)) # Blue
            # 红色画 Fall
            draw_boxes(img_mm_plot, mm_fall_boxes, "FALL", (0, 0, 255))     # Red
            
            # 左上角写诊断
            cv2.putText(img_mm_plot, diagnosis, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(dir_mm, filename), img_mm_plot)

            # --- D. 画图 2: 单模态视图 ---
            img_single_plot = img_raw.copy()
            # 青色画 Single_Person
            draw_boxes(img_single_plot, s_person_boxes, "S_Person", (255, 255, 0)) # Cyan
            # 紫色画 Single_Fall
            draw_boxes(img_single_plot, s_fall_boxes, "S_FALL", (255, 0, 255))     # Magenta
            
            cv2.imwrite(os.path.join(dir_single, filename), img_single_plot)

        # --- E. 记录 ---
        report_data.append({
            "文件名": filename,
            "诊断结论": diagnosis,
            "多模_Fall_Conf": round(mm_fall_conf, 4),
            "多模_Person_Conf": round(mm_person_conf, 4),
            "单模_Fall_Conf": round(s_fall_conf, 4),
            "单模_Person_Conf": round(s_person_conf, 4)
        })

    # 4. 导出
    df = pd.DataFrame(report_data)
    df['sort_key'] = df['诊断结论'].apply(lambda x: 0 if "误识别" in x else (1 if "漏检" in x else 2))
    df = df.sort_values(by=['sort_key', '文件名'])
    df = df.drop(columns=['sort_key'])

    csv_path = os.path.join(OUTPUT_ROOT, "Confusion_Report.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("【完成】")
    print(f"📊 CSV报告: {csv_path}")
    print(f"📂 多模态图: {dir_mm}")
    print(f"📂 单模态图: {dir_single}")
    print("="*50)

if __name__ == "__main__":
    main()