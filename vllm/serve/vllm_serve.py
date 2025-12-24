# run_vllm.py
import argparse
import sys
import json
from pathlib import Path
from backend.command_builder import VLLMCommandBuilder

# 添加项目根到 sys.path，确保 utils 可导入
repo_root = Path(__file__).resolve().parents[2]  # 假设 run_vllm.py 在 infer_tutorial 根目录
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.env_detector import EnvDetector, RuntimeEnv
from utils.util import run_cmd

def main():
    parser = argparse.ArgumentParser(description="启动 VLLM Serve")

    parser.add_argument("--custom-args", type=str, default="{}", help="框架内部自定义参数（JSON）")

    args, extra_params_list = parser.parse_known_args()

    # 解析 custom-args JSON
    try:
        custom_args = json.loads(args.custom_args)
    except json.JSONDecodeError as e:
        print(f"解析 custom-args 失败: {e}")
        return

    try:
        runtime_env: RuntimeEnv = EnvDetector.detect(EnvDetector)
        if "is_mla_model" in custom_args:
            runtime_env.is_mla_model = custom_args.pop("is_mla_model")
        else:
            runtime_env.is_mla_model = True  # 默认值
    except Exception as e:
        print(f"环境检测失败: {str(e)}")
        return
    builder = VLLMCommandBuilder(config_path="./backend/capability_matrix.yaml")

    result = builder.build_command(runtime_env=runtime_env, extra_params_list=extra_params_list)

    print("启动命令:", " ".join(result["command"]))
    


if __name__ == "__main__":
    main()