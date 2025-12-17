import json
import argparse

def load_last_json(json_path):
    """Read last non-empty JSON line."""
    last_line = None
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    return json.loads(last_line) if last_line else None


def load_gpu_log(gpu_log_path):
    """Read last GPU result.log line and convert to dict."""
    last_line = None
    with open(gpu_log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    # result.log 的行是 Python dict 格式，需要 eval
    return eval(last_line) if last_line else None


def merge_and_append(json_path, gpu_data, json_data):
    """If concurrency/input/output match → merge and append new JSON line."""
    keys = ["concurrency", "input_len", "output_len"]
    for k in keys:
        if str(json_data.get(k)) != str(gpu_data.get(k)):
            print("❌ 字段不匹配，不合并")
            return

    # merge
    merged = {
        **json_data, 
        "gpu_nums": gpu_data.get("gpu_nums"),
        "gpu_usage_avg": gpu_data.get("gpu_usage_avg"),
        "temperature_avg": gpu_data.get("temperature_avg"),
        "total_memory_avg": gpu_data.get("total_memory_avg"),
        "used_memory_avg": gpu_data.get("used_memory_avg"),
    }

    merged_line = json.dumps(merged, ensure_ascii=False) 
    # append to json file
    with open(json_path, "rb+") as f:
        f.seek(0, 2)  # 文件末尾
        # 从文件末尾向前找到最后一个换行符
        pos = f.tell() - 1
        while pos > 0:
            f.seek(pos)
            if f.read(1) == b"\n":
                break
            pos -= 1
        f.seek(pos + 1)
        f.truncate()  # 截断最后一行
        f.write(merged_line.encode("utf-8"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to vllm result json file")
    parser.add_argument("--gpu-log", required=True, help="Path to GPU result.log")
    args = parser.parse_args()

    json_last = load_last_json(args.json)
    gpu_last = load_gpu_log(args.gpu_log)

    if not json_last or not gpu_last:
        print("❌ 无法读取 json 或 gpu log")
        exit(1)

    merge_and_append(args.json, gpu_last, json_last)
