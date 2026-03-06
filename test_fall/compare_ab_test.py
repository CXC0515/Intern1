import os
import cv2
import glob
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================= ⚙️ 配置区域 =================
# 待测试图片目录
IMAGE_DIR = r"E:\code\test_fall\跌落检测蹲姿测试\test_images"
# 结果保存主目录
OUTPUT_ROOT = r"E:\code\test_fall\results_AB_test"
# 新增：对比图保存目录
IMG_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "comparison_images") 

# --- 新模型 (多合一) ---
MODEL_NEW_PATH = r"E:\code\test_fall\fall2_m_640.pt"
ID_NEW_FALL = 0
ID_NEW_PERSON = 1
ID_NEW_SQUAT = 2

# --- 旧模型 (单模态) ---
MODEL_OLD_FALL_PATH = r"E:\code\test_fall\models\shuailuo.pt"
MODEL_OLD_PERSON_PATH = r"E:\code\test_fall\models\xingren.pt"

# ⚠️ 阈值调到最低，尽可能暴露所有检测结果
CONF_THRESHOLD = 0.01 

def get_inference(model, image_path):
    """返回完整的推理结果"""
    try:
        return model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)[0]
    except Exception:
        return None

def draw_yolo_style_box(img, box, text, color=(0, 0, 255)):
    """
    完美复刻 YOLO 官方画框风格：自适应粗细、带背景色填充块、白字
    """
    x1, y1, x2, y2 = map(int, box)
    
    # 根据图片尺寸自适应边框粗细
    lw = max(round(sum(img.shape) / 2 * 0.003), 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=lw)
    
    # 字体粗细和缩放比例
    tf = max(lw - 1, 1) 
    font_scale = lw / 3
    
    # 计算文字背景块的大小
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, tf)
    outside = y1 - h >= 3  # 判断框上方是否有足够空间
    
    # 画文字背景色块
    p2 = (x1 + w, y1 - h - 3 if outside else y1 + h + 3)
    cv2.rectangle(img, (x1, y1), p2, color, -1, cv2.LINE_AA)
    
    # 写白色文字
    cv2.putText(img, text, (x1, y1 - 2 if outside else y1 + h + 2), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

def main():
    # 创建所需的文件夹
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

    # 1. 加载模型
    print("⏳ 正在加载新模型 (多合一)...")
    model_new = YOLO(MODEL_NEW_PATH)
    print("⏳ 正在加载旧模型 (跌倒)...")
    model_old_fall = YOLO(MODEL_OLD_FALL_PATH)
    print("⏳ 正在加载旧模型 (行人)...")
    model_old_person = YOLO(MODEL_OLD_PERSON_PATH)

    # 2. 读取图片
    extensions = ['*.jpg', '*.png', '*.jpeg']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    if not image_files:
        print("❌ 没有找到图片！")
        return

    print(f"🚀 开始 A/B 对比测试 (极低阈值 0.01 + 同尺寸大字体)，共 {len(image_files)} 张图片...")
    report_data = []

    # 3. 遍历分析
    for img_path in tqdm(image_files):
        filename = os.path.basename(img_path)
        
        # --- 跑推理 ---
        res_new = get_inference(model_new, img_path)
        res_old_fall = get_inference(model_old_fall, img_path)
        res_old_person = get_inference(model_old_person, img_path)

        if res_new is None:
            continue

        # --- 统计数量 ---
        new_fall, new_person, new_squat = 0, 0, 0
        for box in res_new.boxes:
            cid = int(box.cls[0])
            if cid == ID_NEW_FALL: new_fall += 1
            elif cid == ID_NEW_PERSON: new_person += 1
            elif cid == ID_NEW_SQUAT: new_squat += 1

        old_fall = sum(1 for box in res_old_fall.boxes if int(box.cls[0]) == 0) if res_old_fall else 0
        old_person = sum(1 for box in res_old_person.boxes if int(box.cls[0]) == 0) if res_old_person else 0

        # --- PM 核心业务逻辑诊断 ---
        diagnosis = "⚪ 表现一致"
        if old_fall > 0 and new_fall == 0:
            if new_squat > 0:
                diagnosis = "✅ 优化: 旧模型误报跌倒，新模型识别为蹲姿"
            else:
                diagnosis = "❌ 退步: 旧模型检出跌倒，新模型漏检"
        elif old_fall == 0 and new_fall > 0:
            diagnosis = "🌟 提升: 新模型检出了旧模型漏掉的跌倒"
        elif diagnosis == "⚪ 表现一致":
            if old_person > 0 and new_person == 0:
                diagnosis = "⚠️ 隐患: 新模型漏检行人"
            elif old_person == 0 and new_person > 0:
                diagnosis = "📈 提升: 新模型发现更多行人"

        # --- 🎨 绘制对比拼图 ---
        orig_img = res_new.orig_img
        
        # 动态标题字号计算（保证标题和图比例协调）
        title_scale = max(round(sum(orig_img.shape) / 2 * 0.0015), 1)
        title_thick = max(title_scale, 2)

        # 右侧：新模型结果 (利用 YOLO 自带的绘图)
        img_new_plot = res_new.plot()
        cv2.putText(img_new_plot, "NEW: All-in-One", (15, 45 * title_scale), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 255, 0), title_thick)

        # 左侧：旧模型结果 (利用手写的 YOLO 风格画框函数)
        img_old_plot = orig_img.copy()
        
        # 画旧跌倒 (红框)
        if res_old_fall:
            for box in res_old_fall.boxes:
                if int(box.cls[0]) == 0:
                    label = f"fall {float(box.conf[0]):.2f}"
                    draw_yolo_style_box(img_old_plot, box.xyxy[0], label, color=(0, 0, 255)) # 红色
        
        # 画旧行人 (蓝框)
        if res_old_person:
            for box in res_old_person.boxes:
                if int(box.cls[0]) == 0:
                    label = f"person {float(box.conf[0]):.2f}"
                    draw_yolo_style_box(img_old_plot, box.xyxy[0], label, color=(255, 0, 0)) # 蓝色

        cv2.putText(img_old_plot, "OLD: Fall + Person", (15, 45 * title_scale), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 0, 255), title_thick)

        # 拼接图片：横向拼接 (Left: Old, Right: New)
        combined_img = cv2.hconcat([img_old_plot, img_new_plot])
        
        # 保存拼图
        save_path = os.path.join(IMG_OUTPUT_DIR, f"compare_{filename}")
        cv2.imwrite(save_path, combined_img)

        # 记录数据
        report_data.append({
            "文件名": filename,
            "业务诊断结论": diagnosis,
            "【新】跌倒": new_fall,
            "【旧】跌倒": old_fall,
            "【新】蹲姿": new_squat,
            "【新】行人": new_person,
            "【旧】行人": old_person,
        })

    # 4. 生成报告
    df = pd.DataFrame(report_data)
    df['sort_key'] = df['业务诊断结论'].apply(lambda x: 1 if "⚪" in x else 0)
    df = df.sort_values(by=['sort_key', '文件名']).drop(columns=['sort_key'])

    csv_path = os.path.join(OUTPUT_ROOT, "AB_Test_Report_With_Images.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print("🎉 极限测试完成！(阈值 0.01)")
    print(f"📊 数据报表: {csv_path}")
    print(f"🖼️ 对比拼图: {IMG_OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()