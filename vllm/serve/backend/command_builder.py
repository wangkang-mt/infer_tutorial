import yaml
import re
from typing import Dict, Any, Optional, List
from packaging.version import Version
from pathlib import Path
import sys

# 添加项目根到 sys.path（确保 utils 可导入）
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from utils.env_detector import RuntimeEnv

class CommandBuilder:
    """通用命令构建器基类，支持YAML规则、环境变量、手动参数覆盖"""

    def __init__(self, config_path: str):
        """
        初始化命令构建器
        :param config_path: YAML配置文件路径
        """
        self.config_path = config_path
        self.config_rules = self._load_config()

    def _load_config(self) -> List[Dict[str, Any]]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if isinstance(config, dict):
                return [config]
            elif isinstance(config, list):
                return config
            else:
                raise ValueError("YAML配置格式错误，应为列表或字典")
        except Exception as e:
            raise RuntimeError(f"加载配置失败: {str(e)}")

    def _version_match(self, current_version: str, version_rule: str) -> bool:
        """版本规则匹配（>、>=、<、<=、==、!=）"""
        if not current_version or not version_rule:
            return False
        op_pattern = re.compile(r'^([<>]=?|==|!=|=)?\s*(\d+\.\d+\.\d+.*)$')
        match = op_pattern.match(version_rule.strip())
        if not match:
            return current_version.strip() == version_rule.strip()
        op, target_version = match.groups()
        op = op or "=="
        try:
            current = Version(current_version)
            target = Version(target_version)
        except Exception:
            return False
        if op == ">":
            return current > target
        elif op == ">=":
            return current >= target
        elif op == "<":
            return current < target
        elif op == "<=":
            return current <= target
        elif op in ("==", "="):
            return current == target
        elif op == "!=":
            return current != target
        else:
            return False

    def _match_rule(self, runtime_env: RuntimeEnv) -> Optional[Dict[str, Any]]:
        """根据运行环境匹配规则，子类可覆盖自定义匹配逻辑"""
        for rule in self.config_rules:
            match = rule.get("match", {})
            if match.get("gpu_name") != runtime_env.gpu_name:
                continue
            if match.get("plugin_type") != runtime_env.plugin_type:
                continue
            frame = match.get("frame", {})
            if frame.get("name") != runtime_env.frame_type:
                continue
            if not self._version_match(runtime_env.frame_version, frame.get("version", "")):
                continue
            if hasattr(runtime_env, "is_mla_model") and match.get("is_mla_model") is not None \
                and match.get("is_mla_model") != runtime_env.is_mla_model:
                continue
            return rule
        return None

    def _append_cli_param(self, command: List[str], params: dict):
        """将字典参数转换为CLI参数并追加到命令列表"""
        for k, v in params.items():
            if isinstance(v, bool):
                if v is True:
                    command.append(k)
                continue
            command.extend([k, str(v)])
        
        return command

    def _split_params(self, rule_params: dict, extra_params: dict) -> (dict, dict):
        """分离手动参数和规则参数，手动参数优先"""
        remaining_rule_params = {}

        for k, v in rule_params.items():
            k_param = "--" + k.replace('_', '-')
            if k_param not in extra_params:
                remaining_rule_params[k_param] = v

        return extra_params, remaining_rule_params

    def _collect_env(self, rule_env: dict, kwargs: dict) -> dict:
        """收集环境变量（不参与命令拼接）"""
        env = {}
        env.update(rule_env or {})
        # 如果后续支持手动 env 覆盖，可在这里 merge kwargs
        return env

    def preprocess_extra_params(self, extra_params_list: list) -> dict:
        """预处理额外参数列表为字典"""
        params_dict = {}
        i = 0
        while i < len(extra_params_list):
            params = extra_params_list[i]
            if params.startswith("--"):
                if i + 1 < len(extra_params_list) and not extra_params_list[i + 1].startswith("--"):
                    params_dict[params] = extra_params_list[i + 1]
                    i += 2
                else:
                    params_dict[params] = True
                    i += 1
            elif i == len(extra_params_list) - 1:
                params_dict[params] = True
                i += 1

        return params_dict

    def build_command(self, runtime_env: RuntimeEnv, base_command: str, extra_params_list: list = None, **kwargs) -> Dict[str, Any]:
        """
        构建最终命令
        :param runtime_env: 环境对象
        :param base_command: 基础命令字符串
        :param extra_params_list: 额外的参数列表
        :param kwargs: 手动参数或环境变量
        :return: 字典包含 command 列表、command_str、env 等
        """
        base_command_list = base_command.split()
        extra_params_list = extra_params_list or []

        matched_rule = self._match_rule(runtime_env)
        if not matched_rule:
            final_command = base_command_list + extra_params_list
            return {
                "command": final_command,
                "command_str": " ".join(final_command),
                "env": {},
                "default_rule": None,
            }
        
        rule_params = matched_rule.get("params", {})
        rule_env = matched_rule.get("env", {})

        

        extra_params_dict = self.preprocess_extra_params(extra_params_list or [])

        manual_params, remaining_rule_params = self._split_params(rule_params, extra_params_dict)
        self._append_cli_param(base_command_list, manual_params)
        self._append_cli_param(base_command_list, remaining_rule_params)

        final_command = base_command_list

        return {
            "command": final_command,
            "command_str": " ".join(final_command),
            "env": self._collect_env(rule_env, kwargs),
            "default_rule": matched_rule, 
        }


class VLLMCommandBuilder(CommandBuilder):
    """针对 vllm serve 的命令构建器"""

    def __init__(self, config_path: str):
        super().__init__(config_path)


    def build_command(self, runtime_env: RuntimeEnv, extra_params_list: list = None, **kwargs) -> Dict[str, Any]:
        """固定 base_command 为 'vllm serve'"""
        return super().build_command(runtime_env, base_command="vllm serve", extra_params_list=extra_params_list, **kwargs)



if __name__ == "__main__":
    config_path = "capability_matrix.yaml"
    builder = VLLMCommandBuilder(config_path)
    
    mock_env = RuntimeEnv(
        gpu_name="MTT S5000",
        gpu_count=4,
        gpu_core_count=28,
        plugin_type="vllm_musa",
        plugin_version="0.1.0",
        frame_type="vllm",
        frame_version="0.8.5",
    )
    mock_env.is_mla_model = False

    result = builder.build_command(
        runtime_env=mock_env,
        extra_params_list = ["--model", "/data/models/llama-2-7b", "--trust-remote-code", "--tp", "8", "--block-size", "64"],
    )

    print("启动命令:", result["command_str"])
    print("使用规则:", result["default_rule"])
