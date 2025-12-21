"""
CosyVoice3 语音克隆服务模块
专门用于通过样本音频生成高质量克隆语音的服务

核心功能:
- 基于CosyVoice3的零样本语音克隆
- 高质量语音合成与音频处理
- 参考音频管理与验证
- 性能优化与错误处理
- 模块化设计，易于扩展和维护

设计原则:
- 单一职责原则：每个类只负责一个特定功能
- 开闭原则：对扩展开放，对修改封闭
- 依赖倒置：高层模块不依赖低层模块
- 高内聚低耦合：模块内部功能紧密相关，模块间依赖最小

"""

import os
import sys
import time
import uuid
import json
import logging
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torchaudio
import numpy as np

# 添加CosyVoice路径
COSYVOICE_PATH = Path(__file__).parent.parent / "CosyVoice"
sys.path.append(str(COSYVOICE_PATH))

# 导入现有的工具模块
from .path_manager import PathManager

# CosyVoice导入检查
try:
    from cosyvoice.cli.cosyvoice import AutoModel
    COSYVOICE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"[CosyService] CosyVoice模块导入失败: {e}")
    COSYVOICE_AVAILABLE = False
    AutoModel = None


# ==================== 异常类定义 ====================

class CosyServiceError(Exception):
    """CosyService基础异常类"""
    pass


class ModelLoadError(CosyServiceError):
    """模型加载异常"""
    pass


class AudioValidationError(CosyServiceError):
    """音频验证异常"""
    pass


class VoiceGenerationError(CosyServiceError):
    """语音生成异常"""
    pass


class ConfigurationError(CosyServiceError):
    """配置异常"""
    pass


# ==================== 数据类定义 ====================

@dataclass
class AudioMetadata:
    """音频元数据"""
    file_path: str
    duration: float
    sample_rate: int
    channels: int
    file_size: int
    created_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def is_valid(self) -> bool:
        """检查音频是否有效"""
        return (self.duration > 0 and
                self.sample_rate > 0 and
                os.path.exists(self.file_path))


@dataclass
class VoiceCloneRequest:
    """语音克隆请求"""
    text: str
    reference_audio_path: str
    prompt_text: Optional[str] = None
    output_filename: Optional[str] = None
    speed: float = 1.0
    stream: bool = False
    language: Optional[str] = None

    def __post_init__(self):
        """请求验证"""
        if not self.text or not self.text.strip():
            raise ValueError("文本内容不能为空")

        if len(self.text) > 5000:
            raise ValueError("文本长度不能超过5000字符")

        if not os.path.exists(self.reference_audio_path):
            raise FileNotFoundError(f"参考音频文件不存在: {self.reference_audio_path}")

        if self.speed <= 0 or self.speed > 3.0:
            raise ValueError("语速必须在0-3.0范围内")


@dataclass
class VoiceCloneResult:
    """语音克隆结果"""
    success: bool
    audio_path: Optional[str] = None
    audio_metadata: Optional[AudioMetadata] = None
    generation_time: float = 0.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    error_message: Optional[str] = None
    created_time: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return (self.success and
                self.audio_path and
                os.path.exists(self.audio_path) and
                self.audio_metadata is not None)


# ==================== 枚举类定义 ====================

class AudioFormat(Enum):
    """音频格式枚举"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class GenerationStatus(Enum):
    """生成状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== 工具类 ====================

class LoggerMixin:
    """日志混入类"""

    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        if not hasattr(self, '_logger'):
            self._logger = logging.getLogger(self.__class__.__name__)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)
        return self._logger


class DeviceManager:
    """设备管理器 - 负责计算设备检测和优化"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self.device = self._detect_optimal_device()
            self._initialized = True

    def _detect_optimal_device(self) -> str:
        """检测最优计算设备"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {"device": self.device}

        if self.device == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "cuda_version": torch.version.cuda
            })

        return info


# ==================== 核心功能模块 ====================

class AudioValidator(LoggerMixin):
    """音频验证器 - 负责音频文件的验证和处理"""

    def __init__(self, max_duration: float = 30.0, min_duration: float = 1.0):
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.supported_formats = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}

    def validate_audio_file(self, audio_path: str) -> AudioMetadata:
        """验证音频文件并返回元数据"""
        try:
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                raise AudioValidationError(f"音频文件不存在: {audio_path}")

            # 检查文件格式
            file_ext = Path(audio_path).suffix.lower()
            if file_ext not in self.supported_formats:
                raise AudioValidationError(f"不支持的音频格式: {file_ext}")

            # 获取音频信息
            try:
                # 尝试使用torchaudio.info，如果失败则使用wave等替代方法
                try:
                    info = torchaudio.info(audio_path)
                    sample_rate = info.sample_rate
                    num_frames = info.num_frames
                    num_channels = info.num_channels
                except AttributeError:
                    # 对于较老版本的torchaudio，使用wave模块
                    import wave
                    with wave.open(audio_path, 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        num_frames = wav_file.getnframes()
                        num_channels = wav_file.getnchannels()

                duration = num_frames / sample_rate
                metadata = AudioMetadata(
                    file_path=os.path.abspath(audio_path),
                    duration=duration,
                    sample_rate=sample_rate,
                    channels=num_channels,
                    file_size=os.path.getsize(audio_path)
                )
            except Exception as e:
                raise AudioValidationError(f"音频文件读取失败: {e}")

            # 验证音频时长
            if metadata.duration < self.min_duration:
                raise AudioValidationError(f"音频时长过短: {metadata.duration:.2f}s < {self.min_duration}s")

            if metadata.duration > self.max_duration:
                raise AudioValidationError(f"音频时长过长: {metadata.duration:.2f}s > {self.max_duration}s")

            # 验证采样率
            if metadata.sample_rate < 16000:
                raise AudioValidationError(f"采样率过低: {metadata.sample_rate}Hz < 16000Hz")

            self.logger.info(f"[AudioValidator] 音频验证通过: {audio_path}")
            return metadata

        except AudioValidationError:
            raise
        except Exception as e:
            raise AudioValidationError(f"音频验证失败: {e}")


class ModelManager(LoggerMixin):
    """模型管理器 - 负责CosyVoice3模型的加载和管理"""

    def __init__(self, model_dir: str, device_manager: DeviceManager):
        self.model_dir = model_dir
        self.device_manager = device_manager
        self.model = None
        self.model_info = {}
        self._lock = threading.RLock()
        self._load_model()

    def _load_model(self):
        """加载CosyVoice3模型"""
        if not COSYVOICE_AVAILABLE:
            raise ModelLoadError("CosyVoice模块不可用")

        with self._lock:
            try:
                self.logger.info(f"[ModelManager] 开始加载模型: {self.model_dir}")

                # 检查模型目录
                if not os.path.exists(self.model_dir):
                    raise ModelLoadError(f"模型目录不存在: {self.model_dir}")

                # 加载模型
                self.model = AutoModel(model_dir=self.model_dir)

                # 获取模型信息
                self.model_info = {
                    "model_dir": self.model_dir,
                    "sample_rate": getattr(self.model, 'sample_rate', 24000),
                    "loaded_time": time.time(),
                    "device": self.device_manager.device
                }

                self.logger.info(f"[ModelManager] 模型加载成功")
                self.logger.info(f"  采样率: {self.model_info['sample_rate']}Hz")
                self.logger.info(f"  设备: {self.model_info['device']}")

            except Exception as e:
                self.logger.error(f"[ModelManager] 模型加载失败: {e}")
                raise ModelLoadError(f"模型加载失败: {e}")

    def get_model(self):
        """获取模型实例"""
        with self._lock:
            if self.model is None:
                raise ModelLoadError("模型未加载")
            return self.model

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return self.model_info.copy()

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None


class AudioProcessor(LoggerMixin):
    """音频处理器 - 负责音频文件的加载、保存和格式转换"""

    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """加载音频文件"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # 转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            return waveform, sample_rate

        except Exception as e:
            raise AudioValidationError(f"音频加载失败: {e}")

    def save_audio(self, audio_data: torch.Tensor, sample_rate: int, output_path: str) -> bool:
        """保存音频文件"""
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # 保存音频
            torchaudio.save(output_path, audio_data, sample_rate)

            self.logger.info(f"[AudioProcessor] 音频已保存: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"[AudioProcessor] 音频保存失败: {e}")
            return False

    def get_output_path(self, filename: str = None, format: str = "wav") -> str:
        """生成输出文件路径"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cosyvoice_clone_{timestamp}.{format}"

        return self.path_manager.get_output_voice_path(filename)


class VoiceCloner(LoggerMixin):
    """语音克隆器 - 核心语音克隆逻辑"""

    def __init__(self, model_manager: ModelManager, audio_processor: AudioProcessor):
        self.model_manager = model_manager
        self.audio_processor = audio_processor
        self.executor = ThreadPoolExecutor(max_workers=2)

    def clone_voice(self, request: VoiceCloneRequest) -> VoiceCloneResult:
        """执行语音克隆"""
        start_time = time.time()
        result = VoiceCloneResult(success=False)

        try:
            self.logger.info(f"[VoiceCloner] 开始语音克隆: {request.request_id}")
            self.logger.info(f"  文本长度: {len(request.text)} 字符")
            self.logger.info(f"  参考音频: {request.reference_audio_path}")

            # 获取模型
            model = self.model_manager.get_model()

            # 生成输出路径
            if request.output_filename:
                output_path = self.audio_processor.get_output_path(request.output_filename)
            else:
                output_path = self.audio_processor.get_output_path()

            # 准备提示文本 - 使用CosyVoice3推荐的高级提示格式
            if request.prompt_text:
                prompt_text = request.prompt_text
            else:
                prompt_text = "You are a helpful assistant. Please speak the following content naturally.<|endofprompt|>"

            # 执行语音克隆
            self.logger.info("[VoiceCloner] 开始生成语音...")

            generated = False
            for i, audio_data in enumerate(model.inference_zero_shot(
                request.text,
                prompt_text,
                request.reference_audio_path,
                stream=request.stream,
                speed=request.speed
            )):
                if 'tts_speech' in audio_data:
                    # 保存生成的音频
                    if self.audio_processor.save_audio(
                        audio_data['tts_speech'],
                        self.model_manager.model_info['sample_rate'],
                        output_path
                    ):
                        result = VoiceCloneResult(
                            success=True,
                            audio_path=output_path,
                            generation_time=time.time() - start_time
                        )

                        # 获取音频元数据
                        try:
                            info = torchaudio.info(output_path)
                            result.audio_metadata = AudioMetadata(
                                file_path=output_path,
                                duration=info.num_frames / info.sample_rate,
                                sample_rate=info.sample_rate,
                                channels=info.num_channels,
                                file_size=os.path.getsize(output_path)
                            )
                        except Exception as e:
                            self.logger.warning(f"获取音频元数据失败: {e}")

                        generated = True
                        break

            if not generated:
                raise VoiceGenerationError("语音生成失败：未生成有效音频")

            self.logger.info(f"[VoiceCloner] 语音克隆成功: {output_path}")
            self.logger.info(f"  生成时长: {result.audio_metadata.duration:.2f}s")
            self.logger.info(f"  耗时: {result.generation_time:.2f}s")

            return result

        except Exception as e:
            result.error_message = str(e)
            result.generation_time = time.time() - start_time
            self.logger.error(f"[VoiceCloner] 语音克隆失败: {e}")
            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
            return result

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# ==================== 主服务类 ====================

class CosyService(LoggerMixin):
    """
    CosyVoice3语音克隆服务主类

    提供统一的API接口，整合所有功能模块，实现高质量的语音克隆服务。

    主要功能:
    - 零样本语音克隆
    - 音频文件验证和处理
    - 模型管理和加载
    - 结果输出和管理

    使用示例:
        service = CosyService()
        result = service.clone_voice(
            text="你好，这是测试语音。",
            reference_audio_path="/path/to/reference.wav"
        )
        if result.is_valid:
            print(f"克隆成功: {result.audio_path}")
    """

    def __init__(self, model_dir: str = None):
        if not hasattr(self, '_initialized'):
            self._initialize_service(model_dir)

    def _initialize_service(self, model_dir: str = None):
        """初始化服务组件"""
        try:
            self.logger.info("[CosyService] 初始化CosyVoice3语音克隆服务...")

            # 初始化基础组件
            self.path_manager = PathManager()
            self.device_manager = DeviceManager()
            self.audio_validator = AudioValidator()
            self.audio_processor = AudioProcessor(self.path_manager)

            # 只有在CosyVoice可用时才初始化模型相关组件
            if COSYVOICE_AVAILABLE:
                # 设置默认模型目录
                if model_dir is None:
                    model_dir = self.path_manager.get_root_begin_path(
                        "CosyVoice", "pretrained_models", "Fun-CosyVoice3-0.5B"
                    )

                # 初始化模型相关模块
                self.model_manager = ModelManager(model_dir, self.device_manager)
                self.voice_cloner = VoiceCloner(self.model_manager, self.audio_processor)
            else:
                self.model_manager = None
                self.voice_cloner = None
                self.logger.warning("[CosyService] CosyVoice模块不可用，仅提供基础音频验证功能")

            self._initialized = True

            # 输出初始化信息
            self.logger.info("[CosyService] 服务初始化完成")
            self.logger.info(f"  设备: {self.device_manager.get_device_info()}")

            if self.model_manager:
                self.logger.info(f"  模型: {self.model_manager.get_model_info()}")
            else:
                self.logger.info("  模型: 未加载（CosyVoice不可用）")

        except Exception as e:
            self.logger.error(f"[CosyService] 服务初始化失败: {e}")
            raise CosyServiceError(f"服务初始化失败: {e}")

    def clone_voice(self, text: str, reference_audio_path: str,
                   prompt_text: str = None, output_filename: str = None,
                   speed: float = 1.0, stream: bool = False) -> VoiceCloneResult:
        """
        语音克隆主接口

        Args:
            text: 要克隆的文本内容
            reference_audio_path: 参考音频文件路径
            prompt_text: 提示文本（可选，推荐使用高级提示格式）
            output_filename: 输出文件名（可选）
            speed: 语速控制（0.1-3.0）
            stream: 是否使用流式推理

        Returns:
            VoiceCloneResult: 克隆结果

        Raises:
            CosyServiceError: 服务相关异常
        """
        try:
            # 检查CosyVoice是否可用
            if not COSYVOICE_AVAILABLE or not self.voice_cloner:
                return VoiceCloneResult(
                    success=False,
                    error_message="CosyVoice模块不可用，无法进行语音克隆"
                )

            # 创建请求对象
            request = VoiceCloneRequest(
                text=text,
                reference_audio_path=reference_audio_path,
                prompt_text=prompt_text,
                output_filename=output_filename,
                speed=speed,
                stream=stream
            )

            # 验证参考音频
            self.audio_validator.validate_audio_file(reference_audio_path)

            # 执行语音克隆
            result = self.voice_cloner.clone_voice(request)

            return result

        except Exception as e:
            self.logger.error(f"[CosyService] 语音克隆失败: {e}")
            return VoiceCloneResult(
                success=False,
                error_message=str(e)
            )

    def validate_reference_audio(self, audio_path: str) -> AudioMetadata:
        """
        验证参考音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            AudioMetadata: 音频元数据

        Raises:
            AudioValidationError: 音频验证异常
        """
        return self.audio_validator.validate_audio_file(audio_path)

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        model_loaded = self.model_manager.is_model_loaded() if self.model_manager else False
        model_info = self.model_manager.get_model_info() if self.model_manager else None

        return {
            "service_initialized": self._initialized,
            "cosyvoice_available": COSYVOICE_AVAILABLE,
            "device_info": self.device_manager.get_device_info(),
            "model_info": model_info,
            "model_loaded": model_loaded
        }

    def cleanup(self):
        """清理服务资源"""
        try:
            if hasattr(self, 'voice_cloner'):
                self.voice_cloner.cleanup()

            self.logger.info("[CosyService] 服务资源清理完成")

        except Exception as e:
            self.logger.error(f"[CosyService] 服务清理失败: {e}")


# ==================== 便捷函数 ====================

def get_cosy_service(model_dir: str = None) -> CosyService:
    """获取CosyService实例"""
    return CosyService(model_dir)


def quick_clone(text: str, reference_audio_path: str,
                output_filename: str = None) -> VoiceCloneResult:
    """快速语音克隆"""
    service = get_cosy_service()
    return service.clone_voice(
        text=text,
        reference_audio_path=reference_audio_path,
        output_filename=output_filename
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    try:
        # 获取服务实例
        service = get_cosy_service()

        # 打印服务状态
        status = service.get_service_status()
        print("=" * 50)
        print("CosyVoice3 语音克隆服务状态")
        print("=" * 50)
        print(f"服务初始化: {status['service_initialized']}")
        print(f"CosyVoice可用: {status['cosyvoice_available']}")
        print(f"设备: {status['device_info']['device']}")
        if status['model_info']:
            print(f"模型采样率: {status['model_info']['sample_rate']}Hz")
            print(f"模型目录: {status['model_info']['model_dir']}")
        print("=" * 50)

        # 示例使用（需要实际的音频文件）
        # result = service.clone_voice(
        #     text="你好，这是CosyVoice3语音克隆测试。",
        #     reference_audio_path="/path/to/reference/audio.wav",
        #     prompt_text="You are a helpful assistant. Please speak the following naturally.<|endofprompt|>",
        #     output_filename="test_clone.wav"
        # )
        #
        # if result.is_valid:
        #     print(f"克隆成功!")
        #     print(f"输出文件: {result.audio_path}")
        #     print(f"音频时长: {result.audio_metadata.duration:.2f}秒")
        #     print(f"生成耗时: {result.generation_time:.2f}秒")
        # else:
        #     print(f"克隆失败: {result.error_message}")

        print("\n服务已就绪，可以开始语音克隆！")

    except Exception as e:
        print(f"服务启动失败: {e}")
        traceback.print_exc()