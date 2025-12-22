### 1. 克隆仓库

```bash
git clone https://github.com/3uyuan1ee/TFG_ui.git
cd TFG_ui
git submodule update --init --recursive
```

### 2. 安装CosyVoice环境

**方法1: 使用Conda（推荐）**
```bash
# 创建Conda环境
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice

# 安装依赖（使用阿里云镜像加速）
cd EchOfU/CosyVoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 返回EchOfU目录
cd ..
pip install -r requirements.txt
```

**方法2: 使用现有Python环境**
```bash
cd EchOfU
pip install -r requirements.txt

# 安装CosyVoice依赖
cd CosyVoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

### 3. 下载CosyVoice预训练模型

**自动下载（推荐）**：
```python
from backend.voice_generator import get_voice_service, ServiceConfig

# 创建服务实例（会自动下载必要模型）
config = ServiceConfig(enable_vllm=True)  # 启用VLLM加速
service = get_voice_service(config)
```

**手动下载**：
```python
# modelscope SDK下载（国内推荐）
from modelscope import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice2-0.5B', local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B')
snapshot_download('iic/CosyVoice-300M', local_dir='CosyVoice/pretrained_models/CosyVoice-300M')
snapshot_download('iic/CosyVoice-300M-SFT', local_dir='CosyVoice/pretrained_models/CosyVoice-300M-SFT')
snapshot_download('iic/CosyVoice-300M-Instruct', local_dir='CosyVoice/pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='CosyVoice/pretrained_models/CosyVoice-ttsfrd')

# 可选：安装ttsfrd以获得更好的文本标准化性能
cd CosyVoice/pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
# 注意：ttsfrd包主要支持Linux，macOS可能需要从源码编译
```

**HuggingFace下载（海外用户）**：
```python
from huggingface_hub import snapshot_download

snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/CosyVoice-300M', local_dir='CosyVoice/pretrained_models/CosyVoice-300M')
snapshot_download('FunAudioLLM/CosyVoice-300M-SFT', local_dir='CosyVoice/pretrained_models/CosyVoice-300M-SFT')
snapshot_download('FunAudioLLM/CosyVoice-300M-Instruct', local_dir='CosyVoice/pretrained_models/CosyVoice-300M-Instruct')
snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='CosyVoice/pretrained_models/CosyVoice-ttsfrd')
```

## 基本使用

```python
from backend.CV_clone import CosyService

# 创建服务实例
service = CosyService()

# 语音克隆
result = service.clone_voice(
    text="你好，这是测试语音。",
    reference_audio_path="path/to/reference.wav"
)

if result.is_valid:
    print(f"克隆成功: {result.audio_path}")
else:
    print(f"克隆失败: {result.error_message}")
```

## 高级用法

### VLLM加速（推荐）

VLLM可以显著提升CosyVoice的推理速度，特别适合批量处理和高频使用场景：

```python
from backend.CV_clone import CosyService

# 启用VLLM加速（需要GPU）
service = CosyService(load_vllm=True)

# VLLM + FP16组合优化（最佳性能）
service = CosyService(
    load_vllm=True,
    fp16=True
)

# 完整优化配置（VLLM + TensorRT + FP16）
service = CosyService(
    load_vllm=True,
    load_trt=True,        # 需要TensorRT环境
    fp16=True,
    trt_concurrent=4     # TensorRT并发数
)
```

**VLLM性能对比**：
- 标准模式：~2-5秒/句
- VLLM模式：~0.5-1秒/句（提升3-5倍）
- VLLM + FP16：~0.3-0.8秒/句（再提升20-30%）

### 性能优化选项

```python
from backend.CV_clone import CosyService

# 启用JIT编译（CPU推理优化）
service = CosyService(load_jit=True)

# 启用FP16（减少内存使用）
service = CosyService(fp16=True)

# 启用TensorRT（需要TensorRT环境，Linux）
service = CosyService(load_trt=True, fp16=True, trt_concurrent=4)

# 组合使用多种优化
service = CosyService(
    load_jit=True,
    fp16=True
)
```

### 性能监控

```python
from backend.CV_clone import CosyService

# 创建带VLLM优化的服务
service = CosyService(load_vllm=True)

# 获取性能统计
if service.model_manager:
    perf_stats = service.model_manager.get_performance_stats()
    print(f"推理次数: {perf_stats['inference_count']}")
    print(f"平均推理时间: {perf_stats['average_inference_time']:.2f}秒")

    # 获取优化信息
    opt_info = service.model_manager.get_optimization_info()
    print(f"VLLM启用: {opt_info['vllm_enabled']}")
    print(f"FP16启用: {opt_info['fp16_enabled']}")
    print(f"JIT启用: {opt_info['jit_enabled']}")
    print(f"TensorRT启用: {opt_info['trt_enabled']}")

    # 重置统计信息
    service.model_manager.reset_performance_stats()
```

### 自定义模型路径

```python
from backend.CV_clone import CosyService

# 使用自定义模型路径
service = CosyService(
    model_dir="/path/to/your/model",
    load_vllm=True
)
```

## 故障排除

### 常见编译问题

**1. grpcio编译错误**
```bash
# 解决方案：使用预编译版本
pip install --no-build-isolation grpcio==1.57.0 grpcio-tools==1.57.0

# 或者使用阿里云镜像
pip install grpcio==1.57.0 grpcio-tools==1.57.0 -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

**2. 子模块更新失败**
```bash
# 方案1：使用EchOfU统一配置
cd /path/to/EchOfU
git submodule update --init --recursive

# 方案2：深度克隆减少下载量
git submodule update --init --recursive --depth=1

# 方案3：如果Matcha-TTS编译失败，暂时跳过
cd CosyVoice
git config submodule.third_party/Matcha-TTS.update none
```

**3. macOS编译问题**
```bash
# 安装必要的系统依赖
brew install cmake sox

# 确保Xcode命令行工具已安装
xcode-select --install

# 使用conda环境（推荐）
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
```

**4. 模型下载失败**
```bash
# 使用ModelScope镜像（国内）
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# 设置环境变量使用国内镜像
export MODELSCOPE_CACHE=./tmp_modelscope
```

**5. 内存不足问题**
```python
# 使用较小的模型
config = ServiceConfig(enable_vllm=False)  # 禁用VLLM

# 或使用量化模型
config = ServiceConfig(enable_vllm=False, model_size="300M")
```

## VLLM安装说明

VLLM需要额外的依赖项，请确保：

```bash
# 安装VLLM
pip install vllm>=0.4.0

# 对于CUDA支持（推荐）
pip install vllm[cuda]

# 对于特定CUDA版本
pip install vllm --index-url https://download.pytorch.org/whl/cu118
```

**注意事项**：
- VLLM需要NVIDIA GPU和CUDA支持
- 建议GPU内存 >= 8GB
- 在CPU环境下，VLLM会自动回退到标准模式
- TensorRT需要Linux环境和额外配置

### 快速测试安装

验证CosyVoice是否正确安装：

```python
import sys
sys.path.append('CosyVoice')

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav

    # 测试模型加载
    cosyvoice = CosyVoice('pretrained_models/CosyVoice2-0.5B')
    print("✅ CosyVoice安装成功！")

    # 测试语音合成
    inference_speech = cosyvoice.inference_sft('你好，欢迎使用CosyVoice！', '中文女')
    print(f"✅ 语音合成测试成功，生成音频长度: {len(inference_speech)}")

except Exception as e:
    print(f"❌ CosyVoice安装或测试失败: {e}")
    print("请检查上述故障排除步骤")
```
