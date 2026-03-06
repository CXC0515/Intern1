import pandas as pd
import os
import cv2
from ultralytics import YOLO

# ================= 核心配置区域 =================

# 1. CSV文件路径
CSV_PATH = r"D:\BaiduNetdiskDownload\export_orig_only_20260122_150347\dataset.csv"

# 2. 模型路径
MODEL_PATH = r"E:\code\model_test\person.pt"

# 3. 高级标签配置
# detect_ids: 会被检测并在图上画框的类别
# score_ids:  仅用这些类别的置信度来计算分数（和旧模型对比）
LABEL_CONFIG = {
    # --- 安全帽 (特别关注) ---
    # 逻辑：检测时同时看 head(1) 和 helmet(2)，方便排查误检
    #       但算分时只看 head(1)，因为 CSV 标签是"未佩戴"，我们要看新模型是否也检出了"未佩戴"
    "未佩戴安全帽": {
        "detect_ids": [1, 2], # 同时检测：未戴(1) + 戴了(2)
        "score_ids": [1]      # 只算分：未戴(1)
    },
    
    # --- 其他常规类别 ---
    "人员闯入": {
        "detect_ids": [0], "score_ids": [0]
    },
    "shouji_phone": {
        "detect_ids": [4], "score_ids": [4]
    },
    "shouji_phone_hand": {
        "detect_ids": [4], "score_ids": [4]
    },
    "xiyan_smoking_hand": {
        "detect_ids": [8], "score_ids": [8]
    },
    "xiyan_smoking": {
        "detect_ids": [8], "score_ids": [8]
    },
    "no-safety-belt": {
        "detect_ids": [6], "score_ids": [6]
    },
    "shuailuo_fall_down": {
        "detect_ids": [7], "score_ids": [7]
    },
    "air_breathing": {
        "detect_ids": [3, 5], "score_ids": [3, 5]
    },
    
    # --- 不需要检测的类别 (配置为空) ---
    # 脚本会自动跳过这些行，不会出现在结果CSV中
    "yanwu_smoke": {"detect_ids": [], "score_ids": []},
    "qihuo_fire":  {"detect_ids": [], "score_ids": []},
    "xielou_spill": {"detect_ids": [], "score_ids": []},
    "jiaoshouban_scaffold_floor": {"detect_ids": [], "score_ids": []}
}

# ===========================================

def main():
    print("--- 启动 Top 100 精细化测试 ---")
    
    if not os.path.exists(CSV_PATH) or not os.path.exists(MODEL_PATH):
        print("错误：找不到CSV或模型文件。")
        return

    base_dir = os.path.dirname(CSV_PATH)
    
    # 图片保存目录
    vis_dir = os.path.join(base_dir, "visualized_top100_refined")
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
        print(f"结果图片将保存至: {vis_dir}")

    print("正在读取数据...")
    try:
        df_all = pd.read_csv(CSV_PATH, header=0, names=['Camera', 'Time', 'Labels', 'Orig_Conf', 'RelPath', 'Name'])
        # 只取前100个
        df = df_all.head(100)
    except Exception as e:
        print(f"CSV读取失败: {e}")
        return

    print("正在加载模型...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    results = []
    
    print(f"开始处理前 {len(df)} 条数据 (自动跳过无效标签)...")
    
    processed_count = 0
    
    for idx, row in df.iterrows():
        img_rel = row['RelPath']
        img_full_path = os.path.join(base_dir, img_rel)
        
        # 1. 解析标签并获取配置
        csv_labels = str(row['Labels'])
        current_detect_ids = []
        current_score_ids = []
        
        for l in csv_labels.split(','):
            l = l.strip()
            if l in LABEL_CONFIG:
                cfg = LABEL_CONFIG[l]
                current_detect_ids.extend(cfg['detect_ids'])
                current_score_ids.extend(cfg['score_ids'])
        
        # 2. 彻底过滤：如果这行数据里没有我们需要算分的ID (比如全是 qihuo_fire)，直接跳过
        if not current_score_ids:
            # print(f"跳过无效标签: {csv_labels}")
            continue
            
        processed_count += 1
        
        # 去重
        current_detect_ids = list(set(current_detect_ids))
        current_score_ids = list(set(current_score_ids))
        
        # 获取原置信度
        try:
            orig_conf = float(row['Orig_Conf'])
        except:
            orig_conf = 0.0
            
        record = {
            '图片名': row['Name'],
            'CSV标签': csv_labels,
            '原置信度': orig_conf,
            '新置信度': 0.0,
            '差值': 0.0,
            '检测结果': '未检测'
        }

        # 3. 推理
        if os.path.exists(img_full_path):
            try:
                # 关键点：classes=current_detect_ids
                # 比如测安全帽时，这里会包含 [1, 2]，所以图上会画出 head 和 helmet
                res = model(img_full_path, verbose=False, classes=current_detect_ids, conf=0.01)[0]
                
                max_score_conf = 0.0
                
                if res.boxes:
                    # 遍历检测到的框，计算分数
                    for box in res.boxes:
                        c_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # 关键点：只有属于 score_ids (如 head=1) 的才算分
                        # 哪怕检测到了 helmet (2) 且分数很高，也不会被计入 max_score_conf
                        # 这样我们就能发现"未佩戴"是否漏检
                        if c_id in current_score_ids:
                            if conf > max_score_conf:
                                max_score_conf = conf
                    
                    record['新置信度'] = max_score_conf
                    
                    # 记录一下到底检出了啥 (方便CSV里看)
                    detected_names = []
                    for box in res.boxes:
                        c_id = int(box.cls[0])
                        c_name = model.names[c_id]
                        conf = float(box.conf[0])
                        detected_names.append(f"{c_name}:{conf:.2f}")
                    record['检测结果'] = ", ".join(detected_names)
                else:
                    record['检测结果'] = "空"

                # 4. 画图并保存
                # plot() 会画出所有 detect_ids 里的框
                plot_img = res.plot()
                
                diff = orig_conf - max_score_conf
                # 标记 diff 方便排序，如果 diff 大说明降分严重
                save_name = f"{processed_count:03d}_diff_{diff:.2f}_{row['Name']}"
                save_path = os.path.join(vis_dir, save_name)
                cv2.imwrite(save_path, plot_img)
                
            except Exception as e:
                print(f"推理出错 {row['Name']}: {e}")
                record['检测结果'] = "Error"
        else:
            record['检测结果'] = "文件缺失"
            # 文件缺失也记录，方便排查
            
        record['差值'] = record['原置信度'] - record['新置信度']
        results.append(record)
        
        if processed_count % 10 == 0:
            print(f"已处理有效图片: {processed_count} 张...")
            
        # 如果有效图片处理够了100张（如果原CSV前100张里有很多火灾，可能需要往后读）
        # 这里逻辑是只读了CSV前100行，其中有效的可能少于100。
        # 如果需要严格凑够100张有效图，需要修改循环逻辑。
        # 目前保持处理CSV前100行。

    # 保存 CSV
    out_file = os.path.join(base_dir, 'top100_refined_report.csv')
    res_df = pd.DataFrame(results)
    # 按差值降序，差值越大说明下降越严重
    res_df.sort_values(by='差值', ascending=False, inplace=True)
    res_df.to_csv(out_file, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"测试完成！")
    print(f"1. 有效测试数量: {processed_count} (已过滤 qihuo_fire 等)")
    print(f"2. 结果图片目录: {vis_dir}")
    print(f"3. 详细报表: {out_file}")
    print("="*40)

if __name__ == "__main__":
    main()