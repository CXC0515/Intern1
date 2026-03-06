import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# ==========================================
# 1. 配置绝对路径 (根据你的输入Input修改)
# ==========================================
# 模型路径
MODEL_PATH = r'E:\code\test_edge_enhancement\anquanmao.pt' 

# 测试图片文件夹路径
TEST_IMG_DIR = r'E:\code\test_edge_enhancement\test_images'

# 测试结果保存路径 (会自动创建)
OUTPUT_DIR = r'E:\code\test_edge_enhancement\test_results'

# 置信度阈值 (根据需要微调)
CONF_THRESHOLD = 0.5         

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. 核心算法：边缘增强 (Group B)
# ==========================================
def enhance_image_edges(image, alpha=1.5, beta=-0.5, gamma=0):
    """
    边缘增强函数
    原理：通过高斯模糊后的差分叠加，提升图像的高频细节（边缘）
    """
    try:
        # 高斯模糊
        gaussian_blur = cv2.GaussianBlur(image, (0, 0), 3.0)
        # 加权叠加：原图 * alpha + 模糊图 * beta + gamma
        enhanced = cv2.addWeighted(image, alpha, gaussian_blur, beta, gamma)
        return enhanced
    except Exception as e:
        print(f"图像增强处理出错: {e}")
        return image # 如果出错，返回原图兜底

# ==========================================
# 3. 主推理逻辑
# ==========================================
def main():
    print(f"--- 开始 A/B 测试 ---")
    print(f"1. 加载模型: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[错误] 找不到模型文件，请检查路径: {MODEL_PATH}")
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

    # 支持的图片扩展名
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(valid_exts)]

    if not image_files:
        print("[警告] 文件夹内没有找到支持的图片格式 (.jpg, .png等)。")
        return

    print(f"   共发现 {len(image_files)} 张图片，开始处理...\n")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(TEST_IMG_DIR, img_file)
        
        # 读取原始图片 (处理中文路径问题)
        # cv2.imread 直接读中文路径在Windows下可能失效，改用 numpy 读取再解码
        try:
            original_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        except Exception:
            original_img = None

        if original_img is None:
            print(f"[{i+1}/{len(image_files)}] 读取失败 (可能是损坏或格式不支持): {img_file}")
            continue
            
        print(f"[{i+1}/{len(image_files)}] 正在处理: {img_file}")

        # --------------------------
        # Group A: 对照组 (原始图片)
        # --------------------------
        start_time_a = time.time()
        results_a = model.predict(original_img, conf=CONF_THRESHOLD, verbose=False)
        time_a = (time.time() - start_time_a) * 1000
        
        # 绘制 Group A
        plot_a = results_a[0].plot() 
        cv2.putText(plot_a, "Group A: Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --------------------------
        # Group B: 实验组 (边缘增强)
        # --------------------------
        # 1. 预处理
        enhanced_img = enhance_image_edges(original_img, alpha=2.0, beta=-1.0)
        
        # 2. 推理
        start_time_b = time.time()
        results_b = model.predict(enhanced_img, conf=CONF_THRESHOLD, verbose=False)
        time_b = (time.time() - start_time_b) * 1000
        
        # 3. 绘制 Group B
        plot_b = results_b[0].plot()
        cv2.putText(plot_b, "Group B: Edge Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --------------------------
        # 结果拼接与保存
        # --------------------------
        # 确保两图高度一致以便拼接
        h_a, w_a = plot_a.shape[:2]
        h_b, w_b = plot_b.shape[:2]
        
        if h_a != h_b:
            plot_b = cv2.resize(plot_b, (w_a, h_a))

        comparison = np.hstack((plot_a, plot_b))
        
        save_name = f"compare_{os.path.splitext(img_file)[0]}.jpg"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        
        # 保存支持中文路径
        cv2.imencode('.jpg', comparison)[1].tofile(save_path)
        
        print(f"    Group A 耗时: {time_a:.1f}ms | 目标: {len(results_a[0].boxes)}")
        print(f"    Group B 耗时: {time_b:.1f}ms | 目标: {len(results_b[0].boxes)}")
        print(f"    已保存 -> {save_path}")

    print(f"\n测试全部完成！请打开文件夹查看结果：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()