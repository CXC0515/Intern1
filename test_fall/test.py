import torch
import cv2
import os
import sys
from pathlib import Path
import warnings

# 忽略无关紧要的警告
warnings.filterwarnings("ignore")

# --- 1. 配置区域---
BASE_DIR = Path(__file__).resolve().parent  # E:\code\test_fall
MODEL_PATH = BASE_DIR / "models" / "shuailuo.pt"
SOURCE_IMG_DIR = BASE_DIR / "test_images"
OUTPUT_DIR = BASE_DIR / "test_results"      # 结果保存在这里

# 置信度阈值 (Confidence Threshold)
# 如果漏报多，把这个数改小 (比如 0.25)
# 如果误报多，把这个数改大 (比如 0.6)
CONF_THRESHOLD = 0.1
def run_local_test():
    print(f"🚀 开始本地离线测试...")
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
    print("⏳ 正在加载模型...")
    try:
        from ultralytics import YOLO
        
        # 直接加载模型
        model = YOLO(str(MODEL_PATH))
        # 把阈值存变量，后面推理用
        print(f"✅ 模型加载成功! (识别为 YOLOv8/Ultralytics 模型)")
        
    except ImportError:
        print("❌ 缺少 ultralytics 库。请运行: pip install ultralytics")
        return
    except Exception as e:
        print(f"❌ YOLOv8 方式加载失败: {e}")

    # --- 4. 遍历并推理 ---
    image_files = list(SOURCE_IMG_DIR.glob("*.jpg")) + list(SOURCE_IMG_DIR.glob("*.png"))
    total_imgs = len(image_files)
    
    if total_imgs == 0:
        print("⚠️ 文件夹里没有图片 (.jpg 或 .png)")
        return

    print(f"📸 找到 {total_imgs} 张图片，开始推理...")
    
    detected_count = 0
    
    for i, img_path in enumerate(image_files):
        try:
            results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False) 
            result = results[0] 
            
            # 统计目标
            # result.boxes 包含检测框
            if len(result.boxes) > 0:
                # 遍历每一个检测到的框
                for box in result.boxes:
                    # 获取坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    # 计算宽和高
                    w = x2 - x1
                    h = y2 - y1
                    ratio = w / h  # 宽高比

                    # 打印详细数据分析
                    print(f"  🔍 分析: 置信度={conf:.2f}, 宽={w:.0f}, 高={h:.0f}, 宽高比={ratio:.2f}")
                    
                    # 逻辑：只过滤“极度瘦长”的站立目标
                    # 宽高比 < 0.6 (即 高度 > 宽度 * 1.66) 判定为站立
                    # 宽高比 0.6 ~ 1.0 (一坨的样子) 判定为跌倒（迎面跌倒）
                    # 宽高比 > 1.0 (扁平) 判定为跌倒
                    
                    if ratio < 0.6:  
                        status = "⚠️ 过滤: 站立误报 (极度瘦长)"
                    else:
                        detected_count += 1
                        status = "🔴 发现跌倒"
                        
                        # 额外记录
                        if ratio < 1.0:
                            status += " (疑似迎面/卷曲跌倒)"
                        else:
                            status += " (侧身/扁平跌倒)"
            else:
                status = "🟢 未检测到"

            print(f"[{i+1}/{total_imgs}] {img_path.name} -> {status}")

            # --- 保存图片 ---
            plotted_img = result.plot()
            
            save_path = OUTPUT_DIR / f"result_{img_path.name}"
            cv2.imwrite(str(save_path), plotted_img)
                
        except Exception as e:
            print(f"❌ 处理图片 {img_path.name} 时出错: {e}")

    # --- 5. 总结报告 ---
    print("\n" + "="*30)
    print(f"📊 测试完成总结")
    print(f"📥 输入图片: {total_imgs} 张")
    print(f"🔴 检出跌倒: {detected_count} 张")
    print(f"💾 结果保存至: {OUTPUT_DIR}")
    print("="*30)
    print("💡 建议: 请打开结果文件夹，肉眼对比检测结果是否符合预期。")

if __name__ == "__main__":
    run_local_test()