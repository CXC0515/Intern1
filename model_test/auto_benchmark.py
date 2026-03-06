import pandas as pd
import os
import cv2
import shutil
import sys
from ultralytics import YOLO

# ================= 1. 配置区域 (无需修改，沿用之前) =================

MODEL_PATH = r"E:\code\model_test\person.pt"
CSV_PATH = r"E:\DATA\Images\wrong\20260110000001_20260115000001\false_alarms.csv"
PASS_THRESHOLD = 0.45 

LABEL_MAP = {
    "人员闯入": 0, "xingren_person": 0,
    "未佩戴安全帽": 1, "anquanmao_head": 1, "head": 1,
    "shouji_phone_hand": 4, "phone": 4,
    "xiyan_smoking_hand": 8, "smoking": 8,
    "fall": 7,
    "未佩戴安全带": 6, "safety_belt": 6,
}

# ================= 2. 日志记录器类 (新增部分) =================

class DualLogger(object):
    """
    黑科技小工具：把 print 的内容同时写到 屏幕 和 文件
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保实时写入，防止程序崩溃没保存

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ================= 3. 主逻辑 =================

def main():
    # --- 路径检查 ---
    if not os.path.exists(CSV_PATH):
        print(f"[错误] 找不到 CSV 文件: {CSV_PATH}")
        return
    
    DATA_ROOT = os.path.dirname(CSV_PATH)
    OUTPUT_DIR = os.path.join(DATA_ROOT, "benchmark_results")
    DIR_FIXED = os.path.join(OUTPUT_DIR, "fixed_success")
    DIR_FAIL = os.path.join(OUTPUT_DIR, "still_fail")
    
    # 清理并重建目录
    if os.path.exists(OUTPUT_DIR):
        try: shutil.rmtree(OUTPUT_DIR)
        except: pass
    os.makedirs(DIR_FIXED, exist_ok=True)
    os.makedirs(DIR_FAIL, exist_ok=True)

    # --- 核心：开启日志记录 ---
    # 所有的 print 都会自动存进这个 txt
    log_file_path = os.path.join(OUTPUT_DIR, "console_log.txt")
    sys.stdout = DualLogger(log_file_path)

    print(f"[-] 测试启动！")
    print(f"[-] 日志将同步保存至: {log_file_path}")
    print(f"[-] 正在加载模型: {MODEL_PATH}")
    
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[致命错误] 模型加载失败: {e}")
        return

    try:
        df = pd.read_csv(CSV_PATH)
        print(f"[-] 成功读取 CSV，共 {len(df)} 条样本。")
    except Exception as e:
        print(f"[错误] 读取 CSV 失败: {e}")
        return

    results_log = []
    fixed_count = 0
    total_processed = 0

    print("-" * 60)
    print(f"{'Row':<6} | {'状态':<8} | {'原标签 (Conf)':<30} -> {'新 Conf':<10}")
    print("-" * 60)

    for index, row in df.iterrows():
        rel_path = row.get('原图路径', '')
        if pd.isna(rel_path) or rel_path == '': continue
            
        full_img_path = os.path.join(DATA_ROOT, rel_path)
        old_label = row.get('标签', 'Unknown')
        old_conf = row.get('置信度', 0)

        if not os.path.exists(full_img_path):
            print(f"Row {index:<5} | [跳过] 图片缺失: {rel_path}")
            continue

        total_processed += 1
        
        try:
            results = model.predict(full_img_path, conf=0.01, verbose=False)[0]
        except Exception as e:
            print(f"Row {index:<5} | [错误] 推理失败: {e}")
            continue

        target_class_id = LABEL_MAP.get(old_label)
        is_fixed = True
        new_conf = 0.0
        details = "未检出目标"
        detected_target = False

        if target_class_id is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id == target_class_id:
                    detected_target = True
                    if conf > new_conf: new_conf = conf
            
            if new_conf > PASS_THRESHOLD:
                is_fixed = False
                details = f"依然误报 (Conf: {new_conf:.2f})"
            elif detected_target:
                details = f"已压制 (Conf: {new_conf:.2f})"
            else:
                details = "完全消除 (未检出)"
        else:
            details = f"跳过: 未知标签 {old_label}"

        # 绘图保存
        plotted_img = results.plot(labels=True, conf=True, line_width=2)
        status_text = "FIXED" if is_fixed else "FAIL"
        color = (0, 255, 0) if is_fixed else (0, 0, 255)
        
        cv2.rectangle(plotted_img, (0, 0), (300, 50), color, -1)
        cv2.putText(plotted_img, f"{status_text} (New: {new_conf:.2f})", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        file_name = f"Row{index}_{os.path.basename(rel_path)}"
        save_path = os.path.join(DIR_FIXED if is_fixed else DIR_FAIL, file_name)
        cv2.imwrite(save_path, plotted_img)

        if is_fixed: fixed_count += 1
        
        # 格式化输出，让日志对齐更好看
        icon = '✅ PASS' if is_fixed else '❌ FAIL'
        old_info = f"{old_label}({old_conf})"
        print(f"Row {index:<5} | {icon:<8} | {old_info:<30} -> {new_conf:.2f}")

        results_log.append({
            "Index": index,
            "原图": rel_path,
            "原误报标签": old_label,
            "原置信度": old_conf,
            "新置信度": new_conf,
            "结论": "PASS" if is_fixed else "FAIL",
            "详细": details,
            "结果图": save_path
        })

    # 生成 CSV 和总结
    if results_log:
        pd.DataFrame(results_log).to_csv(os.path.join(OUTPUT_DIR, "benchmark_report.csv"), index=False, encoding='utf-8-sig')
        
        fix_rate = (fixed_count / total_processed * 100) if total_processed > 0 else 0
        
        print("-" * 60)
        print(f"【测试总结】")
        print(f"处理总量: {total_processed}")
        print(f"修复数量: {fixed_count}")
        print(f"修复比率: {fix_rate:.2f}%")
        print(f"结果目录: {OUTPUT_DIR}")
        print(f"终端日志: {log_file_path}")
        print("-" * 60)

if __name__ == "__main__":
    main()