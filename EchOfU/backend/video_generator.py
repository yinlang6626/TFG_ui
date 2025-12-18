import os
import time
import subprocess
import shutil
from .voice_generator import OpenVoiceService


def generate_video(data):
    """
    模拟视频生成逻辑：接收来自前端的参数，并返回一个视频路径。
    """
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    if data['model_name'] == "SyncTalk":
        try:
            
            # 构建命令
            cmd = [
                './SyncTalk/run_synctalk.sh', 'infer',
                '--model_dir', data['model_param'],
                '--audio_path', data['ref_audio'],
                '--gpu', data['gpu_choice']
            ]

            print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")

            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
                # check=True
            )
            
            print("命令标准输出:", result.stdout)
            if result.stderr:
                print("命令标准错误:", result.stderr)
            
            # 文件原路径与目的路径 
            model_dir_name = os.path.basename(data['model_param'])
            source_path = os.path.join("SyncTalk", "model", model_dir_name, "results", "test_audio.mp4")
            audio_name = os.path.splitext(os.path.basename(data['ref_audio']))[0]
            video_filename = f"{model_dir_name}_{audio_name}.mp4"
            destination_path = os.path.join("static", "videos", video_filename)
            # 检查文件是否存在
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"[backend.video_generator] 视频生成完成，路径：{destination_path}")
                return destination_path
            else:
                print(f"[backend.video_generator] 视频文件不存在: {source_path}")
                # 尝试查找任何新生成的mp4文件
                results_dir = os.path.join("SyncTalk", "model", model_dir_name, "results")
                if os.path.exists(results_dir):
                    mp4_files = [f for f in os.listdir(results_dir) if f.endswith('.mp4')]
                    if mp4_files:
                        latest_file = max(mp4_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
                        source_path = os.path.join(results_dir, latest_file)
                        shutil.copy(source_path, destination_path)
                        print(f"[backend.video_generator] 找到最新视频文件: {destination_path}")
                        return destination_path
                
                return os.path.join("static", "videos", "out.mp4")
            
        except subprocess.CalledProcessError as e:
            print(f"[backend.video_generator] 命令执行失败: {e}")
            print("错误输出:", e.stderr)
            return os.path.join("static", "videos", "out.mp4")
        except Exception as e:
            print(f"[backend.video_generator] 其他错误: {e}")
            return os.path.join("static", "videos", "out.mp4")
    elif data['model_name'] == "ER-NeRF":
        try:
            # [Fix] 实例化 OpenVoiceService (缺少括号)
            ov = OpenVoiceService()

            # 获取基本参数
            # speaker_id 用于标识音色特征
            speaker_id = data.get('speaker_id', 'default_speaker')
            text = data.get('text', '')
            ref_audio = data.get('ref_audio', '')
            
            # ToDo : 语音特征提取
            # 如果提供了参考音频，且需要克隆 (这里假设只要有ref_audio就更新特征)
            if ref_audio and os.path.exists(ref_audio):
                print(f"[backend.video_generator] 提取并保存说话人特征: {speaker_id}")
                ov.extract_and_save_speaker_feature(speaker_id, ref_audio)

            # 实现语音克隆 (Text -> Audio)
            # 如果有文本输入，则生成新的语音文件；否则使用原 ref_audio
            if text:
                print(f"[backend.video_generator] 正在生成语音: {text}")
                generated_audio = ov.generate_speech(text, speaker_id)
                if generated_audio:
                    ref_audio = generated_audio
                    print(f"[backend.video_generator] 语音生成成功: {ref_audio}")
            
            # [加分项] 音频变调处理 (Pitch Shift)
            # 检查 data 中是否有 pitch 参数 (例如 2.0 或 -1.5)
            pitch_steps = data.get('pitch')
            if pitch_steps and ref_audio and os.path.exists(ref_audio):
                try:
                    # 在此处局部引入库，避免修改文件头部
                    import librosa
                    import soundfile as sf
                    
                    print(f"[backend.video_generator] 检测到变调参数: {pitch_steps}，正在处理...")
                    # 读取音频
                    y, sr = librosa.load(ref_audio, sr=None)
                    # 变调
                    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(pitch_steps))
                    
                    # 保存覆盖或另存为新文件
                    dir_name = os.path.dirname(ref_audio)
                    file_name = os.path.basename(ref_audio)
                    ref_audio = os.path.join(dir_name, f"pitch_{pitch_steps}_{file_name}")
                    sf.write(ref_audio, y_shifted, sr)
                    print(f"[backend.video_generator] 变调完成: {ref_audio}")
                except Exception as e:
                    print(f"[backend.video_generator] 变调处理失败 (可能未安装 librosa): {e}")

            # Todo : 实现语音->视频 (ER-NeRF 推理)
            print(f"[backend.video_generator] 开始 ER-NeRF 推理...")
            
            er_nerf_root = "./ER-NeRF"
            # model_param 对应训练时的 workspace 名称 (task_id)
            workspace_name = data.get('model_param')
            
            # 推理命令
            # python main.py data/<workspace> --workspace <workspace> --aud <audio> --test
            cmd = [
                "python", os.path.join(er_nerf_root, "main.py"),
                os.path.join("data", workspace_name), # 数据集路径
                "--workspace", workspace_name,
                "--aud", ref_audio,
                "--test", # 推理模式
                "--test_train" # 使用训练集视角渲染 (Talking Portrait 常用)
            ]
            
            # 设置 GPU
            env = os.environ.copy()
            if 'gpu_choice' in data:
                env['CUDA_VISIBLE_DEVICES'] = str(data['gpu_choice'])

            print(f"[backend.video_generator] 执行 ER-NeRF: {' '.join(cmd)}")
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env,
                cwd=er_nerf_root # 切换工作目录以防相对路径错误
            )

            # 文件结果处理 (寻找生成的 MP4)
            # ER-NeRF 结果通常在 results/<workspace>/...
            results_dir = os.path.join(er_nerf_root, "results", workspace_name)
            
            # 目标输出路径
            audio_name = os.path.splitext(os.path.basename(ref_audio))[0]
            video_filename = f"ernerf_{workspace_name}_{audio_name}.mp4"
            destination_path = os.path.join("static", "videos", video_filename)

            if os.path.exists(results_dir):
                # 遍历查找最新的 MP4 文件
                mp4_files = []
                for root, dirs, files in os.walk(results_dir):
                    for f in files:
                        if f.endswith('.mp4'):
                            mp4_files.append(os.path.join(root, f))
                
                if mp4_files:
                    latest_file = max(mp4_files, key=os.path.getctime)
                    shutil.copy(latest_file, destination_path)
                    print(f"[backend.video_generator] 视频生成完成，路径：{destination_path}")
                    return destination_path
            
            print("[backend.video_generator] 未找到生成的视频文件")
            return os.path.join("static", "videos", "out.mp4")

        except subprocess.CalledProcessError as e:
            print(f"[backend.video_generator] 命令执行失败: {e}")
            print("错误输出:", e.stderr)
            return os.path.join("static", "videos", "out.mp4")
        except Exception as e:
            print(f"[backend.video_generator] 其他错误: {e}")
            return os.path.join("static", "videos", "out.mp4")
    
    video_path = os.path.join("static", "videos", "out.mp4")
    print(f"[backend.video_generator] 视频生成完成，路径：{video_path}")
    return video_path

