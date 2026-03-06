import pandas as pd
import os
import cv2
import shutil
import sys
import glob
from ultralytics import YOLO
import time

# ================= 1. 你的配置 =================

# “错题集”总根目录
ROOT_DIR = r"E:\DATA\Images\wrong"

# 模型路径
MODEL_PATH = r"E:\code\model_test\person.pt"

# 判定阈值
PASS_THRESHOLD = 0.45 

# 标签映射表 (保持不变)
LABEL_MAP = {
    "人员闯入": 0, "xingren_person": 0,
    "未佩戴安全帽": 1, "anquanmao_head": 1, "head": 1,
    "shouji_phone_hand": 4, "phone": 4,
    "xiyan_smoking_hand": 8, "smoking": 8,
    "fall": 7,
    "未佩戴安全带": 6, "safety_belt": 6,
}

# ================= 2. 单个文件夹处理逻辑 =================

def process_single_folder(csv_path, model, folder_index, total_folders):
    """
    处理单个 CSV 文件的核心函数
    返回: {stats_dict} 用于汇总
    """
    current_dir = os.path.dirname(csv_path)
    folder_name = os.path.basename(current_dir)
    
    # 准备输出目录 (在该文件夹内部生成 benchmark_results)
    output_dir = os.path.join(current_dir, "benchmark_results")
    dir_fixed = os.path.join(output_dir, "fixed_success")
    dir_fail = os.path.join(output_dir, "still_fail")
    
    if os.path.exists(output_dir):
        try: shutil.rmtree(output_dir)
        except: pass
    os.makedirs(dir_fixed, exist_ok=True)
    os.makedirs(dir_fail, exist_ok=True)

    print(f"\n[进度 {folder_index}/{total_folders}] 正在巡检: {folder_name}")
    print(f"   📂 路径: {current_dir}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   ⚠️ 读取 CSV 失败: {e}")
        return None

    results_log = []
    fixed_count = 0
    total_count = 0

    for index, row in df.iterrows():
        rel_path = row.get('原图路径', '')
        if pd.isna(rel_path) or rel_path == '': continue
        
        # 拼接图片绝对路径
        full_img_path = os.path.join(current_dir, rel_path)
        old_label = row.get('标签', 'Unknown')
        
        if not os.path.exists(full_img_path):
            continue

        total_count += 1
        
        # 推理
        try:
            results = model.predict(full_img_path, conf=0.01, verbose=False)[0]
        except:
            continue

        # 判定
        target_class_id = LABEL_MAP.get(old_label)
        is_fixed = True
        new_conf = 0.0
        
        if target_class_id is not None:
            for box in results.boxes:
                if int(box.cls[0]) == target_class_id:
                    if float(box.conf[0]) > new_conf:
                        new_conf = float(box.conf[0])
            if new_conf > PASS_THRESHOLD:
                is_fixed = False
        
        # 画图保存 (只保存 FAIL 的图以节省空间，FIXED 的可选)
        # 这里为了演示，FAIL 必存，FIXED 稍微画一下
        plotted_img = results.plot(labels=True, conf=True)
        status = "FIXED" if is_fixed else "FAIL"
        color = (0, 255, 0) if is_fixed else (0, 0, 255)
        
        cv2.rectangle(plotted_img, (0, 0), (300, 50), color, -1)
        cv2.putText(plotted_img, f"{status} (New: {new_conf:.2f})", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        file_name = f"Row{index}_{os.path.basename(rel_path)}"
        
        if is_fixed:
            fixed_count += 1
            cv2.imwrite(os.path.join(dir_fixed, file_name), plotted_img) 
        else:
            cv2.imwrite(os.path.join(dir_fail, file_name), plotted_img) 

        results_log.append({
            "原图": rel_path,
            "标签": old_label,
            "新置信度": new_conf,
            "结果": status
        })

    # 保存单次报告
    if results_log:
        pd.DataFrame(results_log).to_csv(os.path.join(output_dir, "report.csv"), index=False, encoding='utf-8-sig')
    
    fix_rate = (fixed_count / total_count * 100) if total_count > 0 else 0
    print(f"   ✅ 完成! 样本: {total_count} | 修复: {fixed_count} | 修复率: {fix_rate:.1f}%")
    
    return {
        "文件夹名": folder_name,
        "样本数": total_count,
        "修复数": fixed_count,
        "失败数": total_count - fixed_count,
        "修复率(%)": round(fix_rate, 2),
        "完整路径": current_dir
    }

# ================= 3. 主程序 =================

def main():
    print(f"[-] 开始批量扫描根目录: {ROOT_DIR}")
    
    # 1. 递归查找所有的 false_alarms.csv
    target_files = []
    for root, dirs, files in os.walk(ROOT_DIR):
        if "false_alarms.csv" in files:
            target_files.append(os.path.join(root, "false_alarms.csv"))
    
    if not target_files:
        print("[!] 未找到任何 false_alarms.csv 文件！")
        return

    print(f"[-] 共发现 {len(target_files)} 个测试集待处理。")
    print("[-] 加载模型中...")
    
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return

    # 2. 循环处理
    master_summary = []
    
    for i, csv_path in enumerate(target_files):
        stats = process_single_folder(csv_path, model, i+1, len(target_files))
        if stats:
            master_summary.append(stats)

    # 3. 生成总榜单
    if master_summary:
        print("\n" + "="*50)
        print("【全量测试总报告】")
        
        summary_df = pd.DataFrame(master_summary)
        
        # 按照修复率倒序排列（先看最差的）
        summary_df = summary_df.sort_values(by="修复率(%)", ascending=True)
        
        # 打印到终端
        print(summary_df[["文件夹名", "样本数", "修复率(%)", "失败数"]].to_string(index=False))
        
        # 保存到根目录
        save_path = os.path.join(ROOT_DIR, "MASTER_TEST_REPORT.csv")
        summary_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print(f"\n[-] 总表已保存至: {save_path}")
        print("[-] 提示: 请重点检查修复率最低的几个文件夹。")
        print("="*50)

if __name__ == "__main__":
    main()