# run_vllm.py
import argparse
import sys
import json
from pathlib import Path
from backend.command_builder import VLLMCommandBuilder
import subprocess
from datetime import datetime
import os

# 添加项目根到 sys.path，确保 utils 可导入
repo_root = Path(__file__).resolve().parents[2]  # 假设 run_vllm.py 在 infer_tutorial 根目录
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from utils.env_detector import EnvDetector, RuntimeEnv

def ensure_vllm_env(target_env: dict):

    # 规则里没要求 VLLM_USE_V1，就什么都不做
    if "VLLM_USE_V1" not in target_env:
        return

    target_value = str(target_env["VLLM_USE_V1"])

    # 当前进程已经是目标状态，直接返回
    if os.environ.get("VLLM_USE_V1") == target_value:
        return

    # 否则：设置 env 并 exec
    new_env = os.environ.copy()
    new_env["VLLM_USE_V1"] = target_value

    os.execvpe(
        sys.executable,
        [sys.executable] + sys.argv,
        new_env,
    )


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

    ensure_vllm_env(result.get("env", {}))

    print("启动命令:", " ".join(result["command"]))
    # print("使用规则:", result["default_rule"])
    print("环境变量:", result.get("env", {}))

    # 日志目录
    log_dir = Path("vllm_serve_log")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件名：模型名_时间戳.log
    model_name = getattr(runtime_env, "model_name", "unknown_model")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{model_name}_{timestamp}.log"

    print(f"日志保存到: {log_path}")

    # 执行命令并实时输出，同时写入日志
    with open(log_path, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            result["command"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        for line in process.stdout:
            print(line, end="")  # 实时输出到终端
            f.write(line)        # 写入日志

        process.wait()
        exit_code = process.returncode

    if exit_code == 0:
        print("VLLM Serve 启动完成。")
    else:
        print(f"VLLM Serve 启动失败，退出码 {exit_code}。")


if __name__ == "__main__":
    main()