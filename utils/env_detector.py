import re
from dataclasses import dataclass
from typing import Literal, Dict

# 支持两种运行方式：作为包导入（推荐）和直接运行脚本（用于调试）
try:
    # 当作为包导入时（例如 `from utils import EnvDetector`）使用相对导入
    from .util import get_package_info, run_cmd
except (ImportError, SystemError):
    # 当直接运行脚本（`python env_detector.py`）时，包上下文不存在，尝试把仓库根加入 sys.path
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # 现在可以使用绝对导入
    from utils.util import get_package_info, run_cmd


@dataclass
class RuntimeEnv:
    # gpu info
    gpu_name: str  
    gpu_count: int
    gpu_core_count: int

    # framework info
    plugin_type: str  
    plugin_version: str
    frame_type: str  
    frame_version: str


class EnvDetector:
    @staticmethod
    def detect(cls) -> RuntimeEnv:
        detector = cls()
        gpu_info = detector._get_gpu_info()
        framework_info = detector._get_framework_info()
        
        return RuntimeEnv(
            gpu_name=gpu_info["gpu_name"],
            gpu_count=gpu_info["gpu_count"],
            gpu_core_count=gpu_info["gpu_core_count"],
            plugin_type=framework_info["plugin_type"],
            plugin_version=framework_info["plugin_version"],
            frame_type=framework_info["frame_type"],
            frame_version=framework_info["frame_version"],
        )
    
    def _get_framework_info(self) -> Dict[str, str]:
        try:
            framework_version = get_package_info("vllm")
            plugin_version = get_package_info("vllm_musa")
        except Exception as e:
            # 如果包不存在，使用默认值
            framework_version = "unknown"
            plugin_version = "unknown"
        
        framework_info = {
            "frame_type": "vllm",
            "frame_version": str(framework_version),
            "plugin_type": "vllm_musa",
            "plugin_version": str(plugin_version),
        }
        return framework_info

    def _get_gpu_info(self) -> Dict[str, str | int]:
        # 注意：这里假设 run_cmd 返回 (stdout, stderr, return_code)
        out, _, code = run_cmd("mthreads-gmi -q  | grep 'Product Name'")

        if code != 0:
            raise RuntimeError("Failed to get GPU info using mthreads-gmi command.")
        
        lines = out.strip().split('\n')

        gpu_count = len(lines)
        gpu_name = lines[0].split(':')[-1].strip() if lines else "unknown"
        gpu_core_count = self._get_gpu_core_count()
        
        gpu_info = {
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "gpu_core_count": gpu_core_count,
        }

        return gpu_info

    def _get_gpu_core_count(self) -> int:
        out, _, code = run_cmd("musaInfo")

        if code != 0:
            raise RuntimeError("Failed to get GPU core count using musaInfo command.")
        
        match = re.search(r'multiProcessorCount:\s+(\d+)', out)
        if match:
            mp_count = int(match.group(1))
            return mp_count // 2
        else:
            raise RuntimeError("Failed to extract multiProcessorCount from musaInfo output.")
        
# Example usage:

if __name__ == "__main__":
    env = EnvDetector.detect(EnvDetector)
    print(env)