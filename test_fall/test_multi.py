import torch
import cv2
import os
from pathlib import Path
import warnings

# 忽略无关紧要的警告
warnings.filterwarnings("ignore")

# --- 1. 配置区域 ---
# 图片目录
SOURCE_IMG_DIR = Path(r"E:\code\test_fall\test_images")
BASE_DIR = SOURCE_IMG_DIR.parent
# ✅ 更新为你指定的模型路径
MODEL_PATH = Path(r"E:\code\test_fall\fall2_m_640.pt")
OUTPUT_DIR = BASE_DIR / "test_results"       # 结果保存在这里

# 置信度阈值 (Confidence Threshold)
CONF_THRESHOLD = 0.25 

# 类别字典映射 (你的新模型类别)
CLASS_NAMES = {
    0: 'fall (跌倒)', 
    1: 'person (站立)', 
    2: 'squat (蹲姿)'
}

def run_local_test():
    print(f"🚀 开始新多类别模型本地测试...")
    print(f"📂 模型路径: {MODEL_PATH}")
    print(f"📂 图片路径: {SOURCE_IMG_DIR}")

    # --- 2. 检查环境 ---
    if not MODEL_PATH.exists():
        print(f"❌ 错误: 找不到模型文件! 请确认文件在: {MODEL_PATH}")
        return
    if not SOURCE_IMG_DIR.exists():
        print(f"❌ 错误: 找不到图片文件夹! 请确认路径: {SOURCE_IMG_DIR}")
        return
    
    # 创建结果输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 3. 加载模型 ---
    print("⏳ 正在加载新模型...")
    try:
        from ultralytics import YOLO
        # 直接加载模型
        model = YOLO(str(MODEL_PATH))
        print(f"✅ 模型加载成功! (包含类别: {model.names})")
        
    except ImportError:
        print("❌ 缺少 ultralytics 库。请运行: pip install ultralytics")
        return
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    # --- 4. 遍历并推理 ---
    image_files = list(SOURCE_IMG_DIR.glob("*.jpg")) + list(SOURCE_IMG_DIR.glob("*.png"))
    total_imgs = len(image_files)
    
    if total_imgs == 0:
        print("⚠️ 文件夹里没有图片 (.jpg 或 .png)")
        return

    print(f"📸 找到 {total_imgs} 张图片，开始推理...")
    
    # 统计数据
    stats = {'fall': 0, 'person': 0, 'squat': 0, 'empty': 0}
    
    for i, img_path in enumerate(image_files):
        try:
            results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False) 
            result = results[0] 
            
            detected_items = []
            
            # 统计目标
            if len(result.boxes) > 0:
                for box in result.boxes:
                    # 获取类别 ID 和 置信度
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    
                    # 映射类别名称
                    if cls_id == 0:
                        stats['fall'] += 1
                        label = "🔴 fall"
                    elif cls_id == 1:
                        stats['person'] += 1
                        label = "🟢 person"
                    elif cls_id == 2:
                        stats['squat'] += 1
                        label = "🟡 squat"
                    else:
                        label = f"⚪ unknown({cls_id})"
                        
                    detected_items.append(f"{label}({conf:.2f})")
                
                status = "检测到: " + " | ".join(detected_items)
            else:
                stats['empty'] += 1
                status = "⚪ 未检测到任何目标"

            print(f"[{i+1}/{total_imgs}] {img_path.name} -> {status}")

            # --- 保存图片 ---
            # result.plot() 会自动把不同类别的框画上不同颜色和标签
            plotted_img = result.plot() 
            save_path = OUTPUT_DIR / f"result_{img_path.name}"
            cv2.imwrite(str(save_path), plotted_img)
                
        except Exception as e:
            print(f"❌ 处理图片 {img_path.name} 时出错: {e}")

    # --- 5. 总结报告 ---
    print("\n" + "="*35)
    print(f"📊 测试完成总结")
    print(f"📥 输入图片总数: {total_imgs} 张")
    print("-" * 35)
    print(f"🔴 共检出 [跌倒 fall] 目标: {stats['fall']} 个")
    print(f"🟢 共检出 [站立 person] 目标: {stats['person']} 个")
    print(f"🟡 共检出 [蹲姿 squat] 目标: {stats['squat']} 个")
    print(f"⚪ 空白/未检出图片数: {stats['empty']} 张")
    print("-" * 35)
    print(f"💾 画框结果已保存至: {OUTPUT_DIR}")
    print("="*35)
    print("💡 建议: 前往结果文件夹，检查模型对 '蹲姿' 和 '跌倒' 的区分能力。")

if __name__ == "__main__":
    run_local_test()