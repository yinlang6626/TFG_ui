#!/usr/bin/env python3
"""
OpenVoice安装测试脚本
"""

import sys
import os

def test_openvoice_import():
    """测试OpenVoice模块导入"""
    try:
        print("正在测试OpenVoice导入...")

        # 测试基本导入
        import openvoice
        print("✓ 基本导入成功")

        # 测试API导入
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter
        print("✓ API模块导入成功")

        # 测试se_extractor导入
        from openvoice import se_extractor
        print("✓ 特征提取器导入成功")

        # 显示模块信息
        print(f"✓ OpenVoice模块位置: {openvoice.__file__}")

        return True

    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_environment():
    """显示环境信息"""
    print("\n=== 环境信息 ===")
    print(f"Python版本: {sys.version}")
    print(f"Python可执行文件: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")

def suppress_warnings():
    """抑制一些常见的警告"""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

def main():
    """主测试函数"""
    print("OpenVoice安装测试")
    print("=" * 30)

    # 抑制警告
    suppress_warnings()

    # 显示环境信息
    test_environment()

    print("\n=== 导入测试 ===")

    # 测试导入
    success = test_openvoice_import()

    if success:
        print("\n🎉 OpenVoice安装成功，所有模块可以正常导入！")
        return 0
    else:
        print("\n❌ OpenVoice导入失败，请检查安装。")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)