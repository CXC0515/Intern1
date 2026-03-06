import os
import shutil
import pandas as pd
import oracledb
import json
from datetime import datetime, timedelta

# ================= 核心配置 =================
DB_CONFIG = {
    "user": "ADMIN",
    "password": "123456",
    # ⚠️ 请确保这里是工厂电脑的真实 IP
    "dsn": "127.0.0.1:1521/ORCLPDB" 
}

# 表名和字段
TABLE_NAME = "EXEVENT"            
COL_TIME = '"timestamp"'          
COL_CAMERA = '"cameracode"'       
COL_LABEL = '"type"'              
COL_DATA = '"zb"'  # 存放 [{"class": "...", "acc": ...}] 的字段

# 路径配置
SOURCE_ORIG_DIR = r"D:\0-rtspspjk\backend\original_images" 
# SOURCE_ALERT_DIR 已移除，不再需要
EXPORT_DIR = r"D:\0-rtspspjk\export_data_test"

# 标签对照表
LABEL_MAP = {
    "anquanmao_head": "未佩戴安全帽", 
    "xingren_person": "人员闯入",
    "anquanmao": "未佩戴安全帽",
    "anquandai": "未佩戴安全带",
    "jiaoshouban": "脚手板违规",
    "xiyan": "吸烟",
    "shouji": "玩手机",
    "huoyan": "火焰",
    "qihuo": "起火",
    "yanwu": "烟雾",
    "shuailuo": "摔倒/高处坠落",
    "xielou": "泄漏",
    "person": "人员"
}

# ================= 工具函数 =================
def get_time_range():
    print("\n" + "="*50)
    print("【第一步】请选择导出时间范围：")
    print("  1. 手动输入")
    print("  2. 最近 24 小时")
    print("  3. 今天 (00:00:00 到现在)")
    print("="*50)
    choice = input("请输入选项: ").strip()
    now = datetime.now()
    if choice == '1':
        s = input("开始 (YYYY-MM-DD HH:MM:SS): ").strip()
        e = input("结束 (YYYY-MM-DD HH:MM:SS): ").strip()
        return s, e
    elif choice == '2':
        s = now - timedelta(hours=24)
        return s.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        s = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return s.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")

def find_file_match(directory, filename_pattern):
    for ext in ['.jpg', '.png', '.jpeg']:
        real_name = filename_pattern + ext
        full_path = os.path.join(directory, real_name)
        if os.path.exists(full_path):
            return full_path, real_name
    return None, None

# ================= 主程序 =================
def main():
    start_str, end_str = get_time_range()
    
    # 创建导出文件夹
    folder_name = f"export_orig_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(EXPORT_DIR, folder_name)
    save_img_dir = os.path.join(save_dir, "images")
    os.makedirs(save_img_dir, exist_ok=True)
    
    print(f"\n[状态] 连接数据库 {DB_CONFIG['dsn']} ...")

    rows = []
    try:
        with oracledb.connect(user=DB_CONFIG["user"], password=DB_CONFIG["password"], dsn=DB_CONFIG["dsn"]) as conn:
            with conn.cursor() as cursor:
                sql = f"""
                    SELECT {COL_TIME}, {COL_CAMERA}, {COL_LABEL}, {COL_DATA}
                    FROM {TABLE_NAME}
                    WHERE {COL_TIME} >= TO_TIMESTAMP(:s, 'YYYY-MM-DD HH24:MI:SS') 
                      AND {COL_TIME} <= TO_TIMESTAMP(:e, 'YYYY-MM-DD HH24:MI:SS')
                """
                cursor.execute(sql, s=start_str, e=end_str)
                cols = [d[0].lower() for d in cursor.description]
                results = cursor.fetchall()
                print(f"[成功] 查到 {len(results)} 条记录。")
                for res in results:
                    rows.append(dict(zip(cols, res)))

    except oracledb.Error as e:
        print(f"\n[错误] 数据库查询失败: {e}")
        return

    if not rows:
        print("未找到数据。")
        return

    print("\n[状态] 开始解析数据并匹配原图...")
    csv_data = []
    
    for i, row in enumerate(rows):
        db_time = row['timestamp']
        cam_code = str(row['cameracode'])
        raw_label = row['type']
        zb_data = row.get('zb')
        
        real_label = raw_label
        confidence = "N/A"
        
        # --- 解析 JSON 获取准确标签和置信度 ---
        try:
            zb_str = ""
            if zb_data and hasattr(zb_data, 'read'):
                zb_str = zb_data.read()
            elif zb_data:
                zb_str = str(zb_data)
            
            if zb_str:
                clean_str = zb_str.replace("None", "null")
                if clean_str.strip().startswith("'"):
                    clean_str = clean_str.replace("'", '"')
                
                data_obj = json.loads(clean_str)
                
                # 解析 [{"class":...}, {"class":...}]
                if isinstance(data_obj, list) and len(data_obj) > 0:
                    found_classes = set()
                    max_conf = 0.0
                    
                    for item in data_obj:
                        cls = item.get('class') or item.get('label')
                        if cls:
                            cn_name = LABEL_MAP.get(cls, cls)
                            found_classes.add(cn_name)
                        
                        # 提取 acc
                        conf = item.get('acc') or item.get('score') or item.get('conf') or 0
                        if isinstance(conf, (int, float)) and conf > max_conf:
                            max_conf = conf
                            
                    if found_classes:
                        real_label = ", ".join(found_classes)
                    if max_conf > 0:
                        confidence = max_conf

                elif isinstance(data_obj, dict):
                    cls = data_obj.get('class') or data_obj.get('label')
                    if cls:
                        real_label = LABEL_MAP.get(cls, cls)
                    conf = data_obj.get('acc') or data_obj.get('score') or data_obj.get('conf')
                    if conf:
                        confidence = conf

        except Exception as e:
            if i < 3: print(f"[解析警告 Row {i}] {e}")
            pass

        # --- 仅匹配原图 ---
        time_str = db_time.strftime("%Y%m%d%H%M%S")
        pattern_orig = f"{time_str}_{cam_code}"

        path_orig, name_orig = find_file_match(SOURCE_ORIG_DIR, pattern_orig)
        
        # 只有找到原图才添加到结果中
        if path_orig:
            # 复制图片
            shutil.copy2(path_orig, os.path.join(save_img_dir, name_orig))
            csv_path_orig = os.path.join("images", name_orig)
            
            csv_data.append({
                "摄像头编号": cam_code,
                "告警时间": str(db_time),
                "标签": real_label,
                "置信度": confidence,
                "原图路径": csv_path_orig,
                "原图文件名": name_orig
            })

    # 生成 CSV
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = os.path.join(save_dir, "dataset.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print("【导出完成】")
        print(f"模式: 仅导出原图 + CSV")
        print(f"文件夹位置: {save_dir}")
        print(f"共导出 {len(csv_data)} 张图片。")
        
        # --- 自动压缩逻辑 ---
        print("-" * 30)
        print("正在打包压缩，请稍候...")
        try:
            # make_archive 会自动在 save_dir 后面加 .zip
            zip_path = shutil.make_archive(save_dir, 'zip', save_dir)
            print(f"✅ 压缩成功！压缩包位置:\n{zip_path}")
        except Exception as e:
            print(f"❌ 压缩失败: {e}")
            
        print("="*50)
    else:
        print("\n[警告] 未匹配到任何原图文件。")

if __name__ == "__main__":
    main()