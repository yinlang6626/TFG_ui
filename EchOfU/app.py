# app.py
from flask import Flask, render_template, request, jsonify, send_file
import os
import json
from datetime import datetime
from backend.video_generator import generate_video
from backend.model_trainer import train_model
from backend.chat_engine import chat_response
import psutil

# 尝试导入 GPUtil，如果失败则提供空实现
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    class GPUtil:
        @staticmethod
        def getGPUs():
            return []

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 确保目录存在
for folder in ['static/uploads', 'static/audios', 'static/videos', 'static/images', 'static/history']:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        try:
            data = {
                "model_name": request.form.get('model_name', 'SyncTalk'),
                "model_param": request.form.get('model_param', ''),
                "ref_audio": request.form.get('ref_audio', ''),
                "gpu_choice": request.form.get('gpu_choice', 'GPU0'),
                "target_text": request.form.get('target_text', '')
            }
            
            video_path = generate_video(data)
            
            # 记录历史
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'model': data['model_name'],
                'params': data,
                'output': video_path
            }
            
            history_file = 'static/history/video_generation.json'
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(history_entry)
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            return jsonify({
                'status': 'success',
                'video_path': video_path,
                'message': '视频生成成功'
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    # GET 请求 - 渲染模板
    try:
        # 获取GPU信息
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            gpu_list = [f"GPU{i}" for i in range(len(gpus))]
        else:
            gpus = []
            gpu_list = ['GPU0']
    except:
        gpu_list = ['GPU0']
    
    # 获取可用模型
    models = ['SyncTalk', 'Wav2Lip', 'SadTalker']
    synctalk_dir = './SyncTalk/model'
    if os.path.exists(synctalk_dir):
        for item in os.listdir(synctalk_dir):
            if os.path.isdir(os.path.join(synctalk_dir, item)):
                models.append(item)
    
    return render_template('video_generation.html', gpus=gpu_list, models=models)

@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        try:
            data = {
                "model_choice": request.form.get('model_choice', 'SyncTalk'),
                "ref_video": request.form.get('ref_video', ''),
                "gpu_choice": request.form.get('gpu_choice', 'GPU0'),
                "epoch": request.form.get('epoch', '100'),
                "custom_params": request.form.get('custom_params', '')
            }
            
            result = train_model(data)
            
            return jsonify({
                'status': 'success',
                'message': '模型训练开始',
                'task_id': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    return render_template('model_training.html')

@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        try:
            data = {
                "model_name": request.form.get('model_name', 'SyncTalk'),
                "model_param": request.form.get('model_param', ''),
                "voice_clone": request.form.get('voice_clone', 'false'),
                "api_choice": request.form.get('api_choice', 'glm-4-plus')
            }
            
            result = chat_response(data)
            
            return jsonify({
                'status': 'success',
                'response': result,
                'message': '对话生成成功'
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    return render_template('chat_system.html')

@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    audio_path = './static/audios/input.wav'
    audio_file.save(audio_path)
    
    return jsonify({'status': 'success', 'message': '音频保存成功', 'path': audio_path})

@app.route('/api/status')
def system_status():
    """获取系统状态"""
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        gpu_info = []
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'temperature': gpu.temperature
                    })
            except Exception as e:
                print(f"获取GPU信息失败: {e}")
        
        return jsonify({
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used,
            'memory_total': memory.total,
            'disk_percent': disk.percent,
            'gpus': gpu_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/history/<type>')
def get_history(type):
    """获取历史记录"""
    history_file = f'static/history/{type}.json'
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                data = json.load(f)
                return jsonify(data)
            except:
                return jsonify([])
    else:
        return jsonify([])

@app.route('/video/<path:filename>')
def serve_video(filename):
    """提供视频文件"""
    video_path = os.path.join('static', 'videos', filename)
    if os.path.exists(video_path):
        return send_file(video_path)
    else:
        return jsonify({'status': 'error', 'message': '视频文件不存在'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')