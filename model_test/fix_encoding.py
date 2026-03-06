import os
import pandas as pd

# 你的数据根目录
ROOT_DIR = r"E:\DATA\Images\wrong"

def fix_csv_encoding():
    count = 0
    print(f"[-] 开始扫描并修复 CSV 乱码: {ROOT_DIR}")
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(ROOT_DIR):
        for file in files:
            # 只处理生成的报告文件 (report.csv 或 MASTER_TEST_REPORT.csv)
            if file == "report.csv" or file == "MASTER_TEST_REPORT.csv":
                file_path = os.path.join(root, file)
                
                try:
                    # 1. 尝试用默认 utf-8 读取 (乱码文件的原始编码)
                    df = pd.read_csv(file_path, encoding='utf-8')
                    
                    # 2. 强制用 utf-8-sig (Excel 专用编码) 写回
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    
                    print(f"✅ 已修复: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"❌ 修复失败: {file_path} | 原因: {e}")

    print(f"\n[-] 全部完成！共修复 {count} 个文件。现在可以用 Excel 打开了。")

if __name__ == "__main__":
    fix_csv_encoding()