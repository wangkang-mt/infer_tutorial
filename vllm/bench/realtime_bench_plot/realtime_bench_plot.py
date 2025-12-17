import argparse
import json
import time
import streamlit as st

from realtime_bench_core import (
    parse_metadata_list,
    read_new_lines,
    process_records,
    build_chart,
    find_best,
)


# ---------------------------
# CLI 参数解析（使用 parse_known_args 以兼容 streamlit）
# ---------------------------
parser = argparse.ArgumentParser(description="VLLM Real-time Throughput Monitor")
parser.add_argument("--json-file", type=str, required=True, help="Path to json log file")
parser.add_argument(
    "--metadata",
    type=str,
    nargs="*",
    default=[],
    help="Extra metadata like tp=8 dtype=bf16 gpu-mem=0.7",
)
known_args, _ = parser.parse_known_args()

JSON_FILE = known_args.json_file
POLL_INTERVAL = 1

# 解析 metadata 键值对
metadata_dict = parse_metadata_list(known_args.metadata)

# 如果有 Streamlit query params（优先级更高），从 query params 读取 metadata
qp = st.query_params
if "metadata" in qp and qp["metadata"]:
    # 支持 ?metadata=tp=8&metadata=dtype=bf16
    metadata_dict = parse_metadata_list(qp.get("metadata"))
# 支持通过 query params 指定 json 文件： ?json-file=/path/to/log
if "json-file" in qp and qp["json-file"]:
    JSON_FILE = qp.get("json-file")[0]


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔥 VLLM 实时吞吐监控 Dashboard")

# 显示 metadata
if metadata_dict:
    md_lines = "\n".join([f"- **{k}**: `{v}`" for k, v in metadata_dict.items()])
    st.markdown(f"""
### ⚙️ 测试元信息（Metadata）
{md_lines}
""")

# session 中存储所有 JSON
if "records" not in st.session_state:
    st.session_state.records = []

placeholder_chart = st.empty()
placeholder_best = st.empty()
placeholder_model = st.empty()


offset = 0


def main_loop():
    global offset
    while True:
        lines, offset = read_new_lines(JSON_FILE, offset)

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
                st.session_state.records.append(js)
            except Exception:
                continue

        if not st.session_state.records:
            time.sleep(POLL_INTERVAL)
            continue

        df = process_records(st.session_state.records)

        # 模型名
        if "model_name" in df.columns:
            model_names = df["model_name"].unique()
            placeholder_model.markdown(f"### 🧩 当前模型：{' | '.join(model_names)}")

        best = find_best(df)
        if best is not None:
            placeholder_best.markdown(f"""
### 🟢 当前最高吞吐组合
- **模型**：`{best['model_name']}`
- **并发**：`{best['concurrency']}`  
- **输入长度**：`{best['input_len']}`  
- **输出长度**：`{best['output_len']}`  
- **吞吐**：`{best['total_token_throughput']:.2f}` token/s  
""")

        chart = build_chart(df)
        placeholder_chart.altair_chart(chart, width="stretch")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main_loop()
