import os
import shutil
import pandas as pd
from flask import Flask, render_template_string, request, jsonify, send_file
import threading
import webbrowser
import time
import sys
import tkinter as tk
from tkinter import filedialog

# ================= 核心配置 =================
app = Flask(__name__)

# 全局变量
class AppState:
    df = None              
    current_idx = 0        
    source_root = ""       
    dest_root = ""         
    total_count = 0        

state = AppState()

# ================= UI (HTML + CSS + JS) =================
HTML_UI = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>误报数据筛选工作台</title>
    <style>
        :root {
            --bg-color: #f0f2f5;
            --card-bg: #ffffff;
            --primary-color: #1890ff;
            --alert-color: #ff4d4f;
            --orig-color: #40a9ff;
            --label-color: #fa8c16; /* 橙色用于显示标签 */
            --text-main: #333;
            --text-sub: #666;
            --border-radius: 12px;
            --shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }

        body {
            margin: 0; padding: 20px;
            font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
            background-color: var(--bg-color); color: var(--text-main);
            height: 100vh; box-sizing: border-box;
            display: flex; flex-direction: column; overflow: hidden;
        }

        /* 顶部控制栏 */
        .header-card {
            background: var(--card-bg); padding: 20px 30px;
            border-radius: var(--border-radius); box-shadow: var(--shadow);
            display: flex; gap: 20px; align-items: flex-end;
            margin-bottom: 20px; z-index: 100; position: relative;
        }

        .input-group { display: flex; flex-direction: column; flex: 1; }
        .input-group label { font-size: 14px; font-weight: 600; color: var(--text-sub); margin-bottom: 8px; }
        .path-row { display: flex; gap: 8px; }
        .path-row input {
            flex: 1; padding: 10px 15px; border: 1px solid #d9d9d9;
            border-radius: 6px; font-size: 14px; transition: all 0.3s;
            outline: none; background-color: #f9f9f9;
        }
        .btn-select {
            padding: 0 15px; background-color: #fff; border: 1px solid #d9d9d9;
            border-radius: 6px; cursor: pointer; color: #555; font-weight: bold;
        }
        .btn-select:hover { background-color: #f0f0f0; border-color: #bbb; }
        .btn-primary {
            background-color: var(--primary-color); color: white; border: none;
            padding: 11px 25px; border-radius: 6px; font-size: 14px; font-weight: 600;
            cursor: pointer; transition: background 0.3s; height: 40px;
        }
        .btn-primary:hover { background-color: #40a9ff; }
        .status-badge {
            margin-left: auto; background: #e6f7ff; color: #1890ff;
            padding: 5px 15px; border-radius: 20px; font-size: 14px; font-weight: bold;
            border: 1px solid #91d5ff; min-width: 120px; text-align: center;
        }

        .workspace { flex: 1; display: flex; gap: 20px; height: 0; position: relative; }

        .image-card {
            flex: 1; background: var(--card-bg); border-radius: var(--border-radius);
            box-shadow: var(--shadow); display: flex; flex-direction: column;
            overflow: hidden; position: relative; border-top: 5px solid transparent;
        }
        .image-card.alert-card { border-top-color: var(--alert-color); }
        .image-card.orig-card { border-top-color: var(--orig-color); }

        .card-header {
            padding: 15px; text-align: center; font-size: 18px; font-weight: bold;
            border-bottom: 1px solid #f0f0f0; display: flex; justify-content: space-between; align-items: center;
        }
        
        .tag-group { display: flex; gap: 8px; }
        .tag { padding: 4px 10px; border-radius: 6px; font-size: 13px; color: white; display: flex; align-items: center;}
        .tag-red { background: var(--alert-color); }
        .tag-blue { background: var(--orig-color); }
        
        /* 新增：标签样式 */
        .tag-label { 
            background: var(--label-color); 
            font-weight: normal; 
            box-shadow: 0 2px 5px rgba(250, 140, 22, 0.4);
        }

        .image-container {
            flex: 1; padding: 20px; display: flex; justify-content: center;
            align-items: center; background-color: #fafafa; overflow: hidden;
        }
        .image-container img {
            max-width: 100%; max-height: 100%; object-fit: contain;
            border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .footer-tips {
            text-align: center; margin-top: 15px; color: var(--text-sub);
            font-size: 14px; background: rgba(255,255,255,0.6);
            padding: 8px; border-radius: 20px;
        }
        .key-badge {
            background: #fff; border: 1px solid #ccc; border-radius: 4px;
            padding: 2px 8px; font-family: monospace; font-weight: bold;
            box-shadow: 0 2px 0 #ccc; margin: 0 4px;
        }

        .overlay {
            position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255,255,255,0.95); display: flex; justify-content: center;
            align-items: center; z-index: 50; font-size: 20px; color: #555;
            flex-direction: column; gap: 15px; border-radius: var(--border-radius);
        }
        .loader {
            border: 4px solid #f3f3f3; border-top: 4px solid var(--primary-color);
            border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        .anim-reject { animation: flashRed 0.4s ease-out; }
        .anim-pass { animation: flashGreen 0.3s ease-out; }
        @keyframes flashRed { 0% { background-color: white; } 50% { background-color: #ffeaea; border-color: red; } 100% { background-color: white; } }
        @keyframes flashGreen { 0% { background-color: white; } 50% { background-color: #f6ffed; border-color: green; } 100% { background-color: white; } }
    </style>
</head>
<body>

    <div class="header-card">
        <div class="input-group">
            <label>📂 源数据文件夹 (包含 dataset.csv)</label>
            <div class="path-row">
                <input type="text" id="srcPath" placeholder="点击右侧按钮选择..." readonly>
                <button class="btn-select" onclick="chooseFolder('srcPath')">选择...</button>
            </div>
        </div>
        <div class="input-group">
            <label>📂 误报保存位置</label>
            <div class="path-row">
                <input type="text" id="dstPath" placeholder="点击右侧按钮选择..." readonly>
                <button class="btn-select" onclick="chooseFolder('dstPath')">选择...</button>
            </div>
        </div>
        <button class="btn-primary" onclick="initData()">🚀 加载数据</button>
        <div class="status-badge" id="progressInfo">等待加载</div>
    </div>

    <div class="workspace" id="workspace">
        <div class="overlay" id="loadingMask">
            <div style="text-align:center;">
                <p>👋 欢迎使用</p>
                <p style="font-size:16px; color:#888;">请点击上方的“选择...”按钮设置路径，然后点击“加载数据”</p>
            </div>
        </div>

        <div class="image-card alert-card" id="cardLeft">
            <div class="card-header">
                <span>告警截图</span>
                <div class="tag-group">
                    <span class="tag tag-label" id="lblType">...</span>
                    <span class="tag tag-red">Alert</span>
                </div>
            </div>
            <div class="image-container"><img id="imgAlert" src="" alt=""></div>
        </div>

        <div class="image-card orig-card" id="cardRight">
            <div class="card-header">
                <span>原始图片</span>
                <span class="tag tag-blue">Original</span>
            </div>
            <div class="image-container"><img id="imgOrig" src="" alt=""></div>
        </div>
    </div>

    <div class="footer-tips">
        快捷键操作： <span class="key-badge">Enter</span> 跳过 (正确告警) &nbsp;&nbsp;|&nbsp;&nbsp; <span class="key-badge">1</span> 标记误报 (保存并下一张)
    </div>

    <script>
        let isReady = false;
        let isBusy = false; 

        // 调用 Python 弹出选择框
        async function chooseFolder(inputId) {
            try {
                const res = await fetch('/api/choose_dir');
                const data = await res.json();
                if (data.path) {
                    document.getElementById(inputId).value = data.path;
                }
            } catch(e) {
                alert("无法调用文件夹选择框，请检查后台是否运行");
            }
        }

        async function initData() {
            const src = document.getElementById('srcPath').value;
            const dst = document.getElementById('dstPath').value;

            if (!src || !dst) { alert("⚠️ 请先点击“选择...”按钮设置路径！"); return; }

            const mask = document.getElementById('loadingMask');
            mask.innerHTML = '<div class="loader"></div><div>正在读取数据...</div>';

            try {
                const res = await fetch('/api/init', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({src: src, dst: dst})
                });
                const data = await res.json();

                if (data.status === 'ok') {
                    isReady = true;
                    mask.style.display = 'none';
                    updateView(data);
                } else {
                    mask.innerHTML = '<div>❌ ' + data.msg + '</div>';
                    alert("加载失败: " + data.msg);
                }
            } catch(e) {
                mask.innerHTML = '<div>❌ 服务器连接失败</div>';
            }
        }

        document.addEventListener('keydown', async (e) => {
            if (!isReady || isBusy) return;
            if (e.key === 'Enter') {
                triggerAnim('anim-pass');
                await doAction('pass');
            } else if (e.key === '1') {
                triggerAnim('anim-reject');
                await doAction('mark_false');
            }
        });

        function triggerAnim(animClass) {
            const left = document.getElementById('cardLeft');
            const right = document.getElementById('cardRight');
            left.classList.remove('anim-pass', 'anim-reject');
            right.classList.remove('anim-pass', 'anim-reject');
            void left.offsetWidth; 
            left.classList.add(animClass);
            right.classList.add(animClass);
        }

        async function doAction(actionType) {
            isBusy = true;
            try {
                const res = await fetch('/api/action', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ action: actionType })
                });
                const data = await res.json();

                if (data.status === 'finished') {
                    document.getElementById('progressInfo').innerText = "✅ 完成";
                    document.getElementById('loadingMask').innerHTML = '<div>🎉 今日任务全部完成！</div>';
                    document.getElementById('loadingMask').style.display = 'flex';
                    isReady = false;
                } else if (data.status === 'ok') {
                    updateView(data);
                }
            } finally {
                setTimeout(() => { isBusy = false; }, 150);
            }
        }

        function updateView(data) {
            const t = new Date().getTime();
            document.getElementById('imgAlert').src = `/api/image?type=alert&file=${encodeURIComponent(data.alert_name)}&t=${t}`;
            document.getElementById('imgOrig').src = `/api/image?type=orig&file=${encodeURIComponent(data.orig_name)}&t=${t}`;
            document.getElementById('progressInfo').innerText = `进度: ${data.current + 1} / ${data.total}`;
            
            // 更新标签
            document.getElementById('lblType').innerText = data.label ? data.label : 'Unknown';
        }
    </script>
</body>
</html>
"""

# ================= 后端逻辑 =================

@app.route('/')
def index():
    return render_template_string(HTML_UI)

@app.route('/api/choose_dir')
def choose_dir():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askdirectory(title="请选择文件夹")
    root.destroy()
    return jsonify({'path': path})

@app.route('/api/init', methods=['POST'])
def init_api():
    data = request.json
    src = data.get('src')
    dst = data.get('dst')

    if not os.path.exists(src): return jsonify({'status': 'err', 'msg': '源路径不存在'})
    csv_path = os.path.join(src, 'dataset.csv')
    if not os.path.exists(csv_path): return jsonify({'status': 'err', 'msg': f'找不到 dataset.csv'})

    try:
        df = pd.read_csv(csv_path)
        if '告警时间' in df.columns:
            df['告警时间'] = pd.to_datetime(df['告警时间'])
            df = df.sort_values(by='告警时间').reset_index(drop=True)
        
        state.df = df
        state.source_root = src
        state.dest_root = dst
        state.current_idx = 0
        state.total_count = len(df)
        os.makedirs(os.path.join(dst, "images"), exist_ok=True)

        return jsonify(get_current_info())
    except Exception as e:
        return jsonify({'status': 'err', 'msg': str(e)})

@app.route('/api/action', methods=['POST'])
def action_api():
    action = request.json.get('action')
    if state.current_idx >= state.total_count: return jsonify({'status': 'finished'})

    if action == 'mark_false':
        try:
            current_row = state.df.iloc[state.current_idx]
            alert_file = current_row.get('告警图文件名(新)')
            orig_file = current_row.get('原图文件名(新)')
            
            if not pd.isna(alert_file) and not pd.isna(orig_file):
                src_img_dir = os.path.join(state.source_root, "images")
                dst_img_dir = os.path.join(state.dest_root, "images")
                src_alert_path = os.path.join(src_img_dir, alert_file)
                src_orig_path = os.path.join(src_img_dir, orig_file)
                
                if os.path.exists(src_alert_path): shutil.copy2(src_alert_path, os.path.join(dst_img_dir, alert_file))
                if os.path.exists(src_orig_path): shutil.copy2(src_orig_path, os.path.join(dst_img_dir, orig_file))

                csv_save_path = os.path.join(state.dest_root, "false_alarms.csv")
                row_df = pd.DataFrame([current_row])
                is_file_exists = os.path.exists(csv_save_path)
                row_df.to_csv(csv_save_path, mode='a', header=not is_file_exists, index=False, encoding='utf-8-sig')
        except Exception as e: print(f"Error: {e}")
    
    state.current_idx += 1
    if state.current_idx >= state.total_count: return jsonify({'status': 'finished'})
    return jsonify(get_current_info())

@app.route('/api/image')
def image_api():
    filename = request.args.get('file')
    if not filename: return "No filename", 404
    images_dir = os.path.join(state.source_root, "images")
    abs_path = os.path.join(images_dir, filename)
    if not os.path.exists(abs_path): return "Image not found", 404
    return send_file(abs_path)

def get_current_info():
    row = state.df.iloc[state.current_idx]
    # 尝试获取标签，优先使用'标签'列，如果没有则尝试'type'列
    label = row.get('标签')
    if pd.isna(label):
        label = row.get('type', 'Unknown')
        
    return {
        'status': 'ok',
        'current': state.current_idx,
        'total': state.total_count,
        'alert_name': str(row.get('告警图文件名(新)', '')),
        'orig_name': str(row.get('原图文件名(新)', '')),
        'label': str(label)
    }

def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()
    app.run(host='127.0.0.1', port=5000, debug=False)