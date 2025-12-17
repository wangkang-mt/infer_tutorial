#!/usr/bin/env bash
# ==========================
# vLLM Benchmarking Script
# ==========================

# 默认参数（可通过 CLI 覆盖）
MODEL_PATH=""
MODEL_NAME=""
PORT=8000
HOST="localhost"
NUM_PROMPTS=""
INPUT_LEN=1024
OUTPUT_LEN=1024
DATASET_NAME="random"
MAX_CONCURRENCY=""

# 打印帮助信息函数
show_help() {
    echo "==============================================="
    echo "🚀 vLLM Benchmark CLI 使用说明"
    echo "-----------------------------------------------"
    echo "必填参数:"
    echo "  --model-path <路径>    模型文件夹路径，例如: /home/dist/DeepSeek-Coder-V2-Lite-Instruct"
    echo "  --model-name <名称>    模型服务名，例如: deepseek"
    echo ""
    echo "可选参数:"
    echo "  --port <端口>          默认: 8000"
    echo "  --host <主机>          默认: localhost"
    echo "  --max-concurrency <数量> 最大并发数 (默认: 不限制)"
    echo "  --num-prompts <数量>   请求数量 (默认: 1000)"
    echo "  --input-len <长度>     输入长度 (默认: 1024)"
    echo "  --output-len <长度>    输出长度 (默认: 1024)"
    echo "  --dataset <名称>       数据集 (默认: random, 当前仅支持 random)"
    echo "  --extra args...      其他传递给 vllm bench serve 的参数"
    echo ""
    echo "注意: 测试命令默认包含 --ignore-eos 参数。"
    echo "示例:"
    echo "  bash vllm_bench.sh --model-path /home/dist/DeepSeek-Coder-V2-Lite-Instruct --model-name deepseek"
    echo "==============================================="
}

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift ;;
        --model-name) MODEL_NAME="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --host) HOST="$2"; shift ;;
        --max-concurrency) MAX_CONCURRENCY="$2"; shift ;;
        --num-prompts) NUM_PROMPTS="$2"; shift ;;
        --input-len) INPUT_LEN="$2"; shift ;;
        --output-len) OUTPUT_LEN="$2"; shift ;;
        --dataset) DATASET_NAME="$2"; shift ;;
        --extra) shift
                 while [[ "$#" -gt 0 ]]; do
                     EXTRA_ARGS+=("$1")
                     shift
                 done
                 ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "❌ 未知参数: $1"; show_help; exit 1 ;;
    esac
    shift
done

# 检查必填参数
if [[ -z "$MODEL_PATH" || -z "$MODEL_NAME" ]]; then
    echo "❌ 错误: --model-path 和 --model-name 均为必填参数！"
    show_help
    exit 1
fi


# 构建命令数组
CMD=(
  vllm bench serve
  --backend vllm
  --model "$MODEL_PATH"
  --trust-remote-code
  --served-model-name "$MODEL_NAME"
  --dataset-name "$DATASET_NAME"
  --percentile-metrics "ttft,tpot,itl,e2el"
  --metric-percentiles "99"
  --host "$HOST"
  --port "$PORT"
  --random-input-len "$INPUT_LEN"
  --random-output-len "$OUTPUT_LEN"
  --ignore-eos
  "${EXTRA_ARGS[@]}"
)

if [[ -n "$MAX_CONCURRENCY" ]]; then CMD+=(--max-concurrency "$MAX_CONCURRENCY"); fi
if [[ -n "$NUM_PROMPTS" ]]; then CMD+=(--num-prompts "$NUM_PROMPTS"); fi


echo "🚀 启动 vLLM benchmark"
echo "----------------------------"
echo "Model Path:    $MODEL_PATH"
echo "Model Name:    $MODEL_NAME"
echo "Host:          $HOST"
echo "Port:          $PORT"
echo "Dataset:       $DATASET_NAME"
echo "Num Prompts:   ${NUM_PROMPTS:-1000}"
echo "Max Concurrency: ${MAX_CONCURRENCY:-None}"
echo "Input Len:     $INPUT_LEN"
echo "Output Len:    $OUTPUT_LEN"
echo "----------------------------"
echo "执行命令:"
echo "${CMD[@]}"

"${CMD[@]}" 