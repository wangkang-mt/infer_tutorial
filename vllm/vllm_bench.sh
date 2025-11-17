#!/usr/bin/env bash
# ==========================
# vLLM Benchmarking Script
# ==========================

# 默认参数（可通过 CLI 覆盖）
MODEL_PATH=""
MODEL_NAME=""
PORT=8000
HOST="localhost"
REQUEST_RATE=32
NUM_PROMPTS=32
INPUT_LEN=1024
OUTPUT_LEN=1024
BURSTINESS=100
DATASET_NAME="random"

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
    echo "  --request-rate <速率>  每秒请求数 (默认: 32)"
    echo "  --num-prompts <数量>   请求数量 (默认: 32)"
    echo "  --input-len <长度>     输入长度 (默认: 1024)"
    echo "  --output-len <长度>    输出长度 (默认: 1024)"
    echo "  --burstiness <值>      突发度 (默认: 100)"
    echo "  --dataset <名称>       数据集 (默认: random)"
    echo "  --extra args...      其他传递给 vllm bench serve 的参数"
    echo ""
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
        --request-rate) REQUEST_RATE="$2"; shift ;;
        --num-prompts) NUM_PROMPTS="$2"; shift ;;
        --input-len) INPUT_LEN="$2"; shift ;;
        --output-len) OUTPUT_LEN="$2"; shift ;;
        --burstiness) BURSTINESS="$2"; shift ;;
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

# 打印配置信息
echo "🚀 启动 vLLM benchmark"
echo "----------------------------"
echo "Model Path:    $MODEL_PATH"
echo "Model Name:    $MODEL_NAME"
echo "Host:          $HOST"
echo "Port:          $PORT"
echo "Dataset:       $DATASET_NAME"
echo "Request Rate:  $REQUEST_RATE"
echo "Num Prompts:   $NUM_PROMPTS"
echo "Input Len:     $INPUT_LEN"
echo "Output Len:    $OUTPUT_LEN"
echo "Burstiness:    $BURSTINESS"
echo "----------------------------"

# 生成随机种子
SEED=$(date +%s)

# 运行 benchmark
# vllm bench serve \
#   --backend vllm \
#   --model "$MODEL_PATH" \
#   --served-model-name "$MODEL_NAME" \
#   --dataset-name "$DATASET_NAME" \
#   --ignore-eos \
#   --burstiness "$BURSTINESS" \
#   --seed "$SEED" \
#   --trust-remote-code \
#   --percentile-metrics "ttft,tpot,itl,e2el" \
#   --metric-percentiles "99" \
#   --host "$HOST" \
#   --port "$PORT" \
#   --num-prompts "$NUM_PROMPTS" \
#   --request-rate "$REQUEST_RATE" \
#   --random-input-len "$INPUT_LEN" \
#   --random-output-len "$OUTPUT_LEN" \
#     "${EXTRA_ARGS[@]}"


# 构建命令数组
CMD=(
  vllm bench serve
  --backend vllm
  --model "$MODEL_PATH"
  --served-model-name "$MODEL_NAME"
  --dataset-name "$DATASET_NAME"
  --ignore-eos
  --burstiness "$BURSTINESS"
  --seed "$SEED"
  --trust-remote-code
  --percentile-metrics "ttft,tpot,itl,e2el"
  --metric-percentiles "99"
  --host "$HOST"
  --port "$PORT"
  --num-prompts "$NUM_PROMPTS"
  --request-rate "$REQUEST_RATE"
  --random-input-len "$INPUT_LEN"
  --random-output-len "$OUTPUT_LEN"
  "${EXTRA_ARGS[@]}"
)


# 执行命令
"${CMD[@]}" 