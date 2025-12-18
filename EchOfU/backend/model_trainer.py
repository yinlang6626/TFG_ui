import os
import subprocess
import shutil

def train_model(data):
    """
    模拟模型训练逻辑。
    """
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    
     # 路径配置
    ref_video_path = data['ref_video']
    model_choice = data['model_choice']
    
    # 获取任务ID (优先使用 speaker_id，否则使用文件名)
    if data.get('speaker_id'):
        task_id = data['speaker_id']
    else:
        task_id = os.path.splitext(os.path.basename(ref_video_path))[0]

    # 1. 统一模型保存路径: TFG_ui/EchOfU/models/ER-NeRF/
    models_root = os.path.join("models", "ER-NeRF")
    os.makedirs(models_root, exist_ok=True)
    
    # 模型的具体保存位置 (Workspace)
    model_save_path = os.path.join(models_root, task_id)

    print(f"[backend.model_trainer] 任务ID: {task_id}, 目标路径: {model_save_path}")
    print("[backend.model_trainer] 模型训练中...")

    if model_choice == "SyncTalk":
        # SyncTalk 逻辑 (微调路径)
        try:
            cmd = [
                "./SyncTalk/run_synctalk.sh", "train",
                "--video_path", ref_video_path,
                "--gpu", data['gpu_choice'],
                "--epochs", str(data.get('epoch', 10))
            ]
            print(f"[backend.model_trainer] 执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("[backend.model_trainer] SyncTalk 训练输出:", result.stdout)

        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] SyncTalk 训练失败: {e.stderr}")
            return ref_video_path
        except Exception as e:
            print(f"[backend.model_trainer] SyncTalk 错误: {e}")
            return ref_video_path

    elif model_choice == "ER-NeRF":
        try:
            print("[backend.model_trainer] 开始 ER-NeRF 训练流程...")
            
            er_nerf_root = "./ER-NeRF"
            # 预处理数据存放路径 (放在 ER-NeRF 下的 data 目录方便脚本调用)
            preprocess_data_path = os.path.join("data", task_id) 
            
            # 步骤 1: 数据预处理
            print(f"[backend.model_trainer] [1/2] 正在进行数据预处理: {ref_video_path}")
            
            process_cmd = [
                "python", os.path.join(er_nerf_root, "data_utils", "process.py"),
                ref_video_path,
                "--task", task_id
            ]
            subprocess.run(process_cmd, check=True)
            
            # 步骤 2: 模型训练
            print(f"[backend.model_trainer] [2/2] 开始训练 ER-NeRF 模型...")
            
            # 获取训练轮数，默认为10 (前端文档)
            epochs = str(data.get('epoch', 10))
            
            train_cmd = [
                "python", os.path.join(er_nerf_root, "main.py"),
                preprocess_data_path,
                "--workspace", model_save_path,  # 指定统一的模型保存路径
                "-O",
                "--iters", epochs,
                "--save_latest"
            ]

            # 处理自定义参数 (custom_params)
            # 前端格式示例: "lr=0.001" -> 解析为 "--lr 0.001"
            custom_params = data.get('custom_params', '')
            if custom_params:
                print(f"[backend.model_trainer] 解析自定义参数: {custom_params}")
                params_list = custom_params.split(',') # 逗号分隔
                for param in params_list:
                    if '=' in param:
                        key, value = param.split('=')
                        train_cmd.append(f"--{key.strip()}")
                        train_cmd.append(value.strip())

            # GPU 设置
            env = os.environ.copy()
            if 'gpu_choice' in data:
                # 假设 gpu_choice 格式为 "GPU0" -> "0"
                gpu_id = data['gpu_choice'].replace("GPU", "")
                env['CUDA_VISIBLE_DEVICES'] = gpu_id

            print(f"[backend.model_trainer] 执行训练命令: {' '.join(train_cmd)}")
            
            result = subprocess.run(
                train_cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env
            )
            
            # 训练完成后，返回一个演示视频路径或模型路径
            # ER-NeRF 并不直接生成视频作为训练结果，这里返回输入的参考视频作为占位，
            # 或者如果生成了验证视频，可以复制到 res_videos
            print(f"[backend.model_trainer] ER-NeRF 训练成功！模型保存在: {model_save_path}")
            
            # 如果训练过程生成了 validation 视频，可以拷贝一份到 res_videos
            
            
        except subprocess.CalledProcessError as e:
            print(f"[backend.model_trainer] ER-NeRF 训练失败: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            return ref_video_path
        except Exception as e:
            print(f"[backend.model_trainer] 未知错误: {e}")
            return ref_video_path

    print("[backend.model_trainer] 训练流程结束")
    return ref_video_path


#  我发现老师的训练逻辑是单独训练视频生成模型，然后再单独提取语音特征进行克隆，所以在模型训练这里应该不用再提取语音特征




