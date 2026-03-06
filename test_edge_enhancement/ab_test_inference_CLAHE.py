import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# ==========================================
# 1. 配置路径 (保持 _CLAHE 后缀)
# ==========================================
# 模型路径
MODEL_PATH = r'E:\code\test_edge_enhancement\anquanmao.pt' 

# 测试图片文件夹路径
TEST_IMG_DIR = r'E:\code\test_edge_enhancement\test_images'

# 输出路径
OUTPUT_DIR = r'E:\code\test_edge_enhancement\test_results_CLAHE'

# 置信度阈值
CONF_THRESHOLD = 0.5         

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 核心算法：CLAHE (自适应直方图均衡化)
# ==========================================
def enhance_image_clahe(image):
    """
    CLAHE 增强算法
    原理：在 LAB 色彩空间中，仅对 L (亮度) 通道进行自适应直方图均衡化。
    """
    try:
        # 1. 转换颜色空间 BGR -> LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB) 
        # 2. 分离通道
        l, a, b = cv2.split(lab)
        # 3. 创建 CLAHE 对象
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # 4. 对 L 通道应用 CLAHE
        cl = clahe.apply(l)
        # 5. 合并通道
        limg = cv2.merge((cl, a, b))
        # 6. 转回 BGR 空间
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return enhanced
    except Exception as e:
        print(f"CLAHE 处理出错: {e}")
        return image

# ==========================================
# 3. 主推理逻辑
# ==========================================
def main():
    print(f"--- 开始 CLAHE 专项 A/B 测试 (含耗时对比) ---")
    print(f"1. 加载模型: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型文件: {MODEL_PATH}")
        return

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return

    print(f"2. 扫描图片目录: {TEST_IMG_DIR}")
    if not os.path.exists(TEST_IMG_DIR):
        print(f"[错误] 找不到图片目录: {TEST_IMG_DIR}")
        return

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(valid_exts)]

    if not image_files:
        print("[警告] 文件夹内没有图片。")
        return

    print(f"   共发现 {len(image_files)} 张图片，开始处理...\n")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        
        # 读取原始图片
        try:
            original_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        except Exception:
            original_img = None

        if original_img is None:
            continue
            
        print(f"[{i+1}/{len(image_files)}] 正在处理: {img_file}")

        # --------------------------
        # Group A: 对照组 (原始图片)
        # --------------------------
        start_time_a = time.time() # 【新增】开始计时
        results_a = model.predict(original_img, conf=CONF_THRESHOLD, verbose=False)
        time_a = (time.time() - start_time_a) * 1000 # 【新增】计算耗时
        
        plot_a = results_a[0].plot() 
        cv2.putText(plot_a, "Group A: Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --------------------------
        # Group B: 实验组 (CLAHE 增强)
        # --------------------------
        # 1. 预处理
        enhanced_img = enhance_image_clahe(original_img)
        
        # 2. 推理
        start_time_b = time.time() # 开始计时
        results_b = model.predict(enhanced_img, conf=CONF_THRESHOLD, verbose=False)
        time_b = (time.time() - start_time_b) * 1000 # 计算耗时
        
        # 3. 绘制 Group B
        plot_b = results_b[0].plot()
        cv2.putText(plot_b, "Group B: CLAHE Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # --------------------------
        # 结果拼接与保存
        # --------------------------
        h_a, w_a = plot_a.shape[:2]
        h_b, w_b = plot_b.shape[:2]
        if h_a != h_b:
            plot_b = cv2.resize(plot_b, (w_a, h_a))

        comparison = np.hstack((plot_a, plot_b))
        
        save_name = f"compare_CLAHE_{os.path.splitext(img_file)[0]}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        cv2.imencode('.jpg', comparison)[1].tofile(save_path)
        
        # 【修改】打印格式完全对齐 USM 代码
        print(f"    Group A 耗时: {time_a:.1f}ms | 目标: {len(results_a[0].boxes)}")
        print(f"    Group B 耗时: {time_b:.1f}ms | 目标: {len(results_b[0].boxes)}")
        print(f"    已保存 -> {save_path}")

    print(f"\n测试完成！请查看结果文件夹：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()