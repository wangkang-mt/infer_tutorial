#!/usr/bin/env python3
import yaml
import json
import sys
from typing import List, Dict


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calc_request_num(concurrency: int, mode: str) -> int:
    """
    根据 request_num 规则计算请求数
    """
    if mode == "auto":
        return concurrency
    if mode.startswith("x"):
        return concurrency * int(mode[1:])
    raise ValueError(f"Unsupported request_num mode: {mode}")


def build_tasks(cfg: dict) -> List[Dict]:
    mode = cfg.get("mode")
    if mode not in ("grid", "custom"):
        raise ValueError(f"Invalid mode: {mode}")

    request_mode = cfg.get("request_num", "auto")
    tasks = []

    if mode == "grid":
        scenes = cfg.get("scenes")
        concurrency_list = cfg.get("concurrency")

        if not scenes or not concurrency_list:
            raise ValueError("grid mode requires scenes and concurrency")

        for scene in scenes:
            if len(scene) != 2:
                raise ValueError(f"Invalid scene format in grid mode: {scene}")

            input_len, output_len = scene

            for conc in concurrency_list:
                tasks.append({
                    "task_mode": mode,
                    "input_len": input_len,
                    "output_len": output_len,
                    "concurrency": conc,
                    "num_requests": calc_request_num(conc, request_mode),
                })

    elif mode == "custom":
        scenes = cfg.get("scenes")
        if not scenes:
            raise ValueError("custom mode requires scenes")

        for scene in scenes:
            if len(scene) != 3:
                raise ValueError(
                    f"Invalid scene format in custom mode (expect [in, out, conc]): {scene}"
                )

            input_len, output_len, conc = scene
            tasks.append({
                "task_mode": mode,
                "input_len": input_len,
                "output_len": output_len,
                "concurrency": conc,
                "num_requests": calc_request_num(conc, request_mode),
            })

    return tasks


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} bench.cfg", file=sys.stderr)
        sys.exit(1)

    cfg_path = sys.argv[1]
    cfg = load_cfg(cfg_path)
    tasks = build_tasks(cfg)

    print(json.dumps(tasks, indent=2))


if __name__ == "__main__":
    main()
