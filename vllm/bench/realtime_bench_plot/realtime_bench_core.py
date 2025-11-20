import json
import time
import pandas as pd
import altair as alt


def parse_metadata_list(metadata_list):
    """Parse list like ['tp=8', 'dtype=bf16'] into dict."""
    md = {}
    for kv in metadata_list or []:
        if "=" in kv:
            k, v = kv.split("=", 1)
            md[k] = v
    return md


def read_new_lines(path, offset):
    with open(path, "r", encoding="utf-8") as f:
        f.seek(offset)
        lines = f.readlines()
        new_offset = f.tell()
    return lines, new_offset


def build_chart(df):
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="concurrency:Q",
            y="total_token_throughput:Q",
            color="label:N",
            tooltip=[
                "model_name",
                "concurrency",
                "input_len",
                "output_len",
                "total_token_throughput",
            ],
        )
        .properties(height=400)
    )
    return chart


def process_records(records):
    df = pd.DataFrame(records)
    # Ensure dtypes
    if df.empty:
        return df
    for col in ["concurrency", "input_len", "output_len"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    if "total_token_throughput" in df.columns:
        df["total_token_throughput"] = df["total_token_throughput"].astype(float)

    df["label"] = df.apply(lambda r: f"in_{r['input_len']}_out_{r['output_len']}", axis=1)
    return df


def find_best(df):
    if df.empty:
        return None
    if "total_token_throughput" not in df.columns:
        return None
    best = df.loc[df["total_token_throughput"].idxmax()]
    return best
