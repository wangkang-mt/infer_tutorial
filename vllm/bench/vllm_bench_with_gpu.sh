#!/usr/bin/env bash
# ==========================================
# vLLM 批量性能测试脚本
# 作者: kang.wang-ext
# ==========================================

set -e

show_help() {
cat << EOF
用法：
  bash $0 [参数...]

参数列表：
  --model-path PATH        （必填）模型路径
  --gpu-num N              （必填）推理用 GPU 数量，用于生成日志目录以及 GPU util 监控
  --config-file PATH      （可选）测试配置文件路径，默认: config/vllm_bench_config.txt
  --model-name NAME        模型服务名，缺省则自动从路径推断（不做大小写转换）
  --dtype xxx              推理精度，仅用于生成日志标记
  --port PORT              默认: 8000
  --host HOST              默认: localhost
示例：
  bash $0 \\
    --model-path /home/dist/DeepSeek-Coder-V2-Lite-Instruct \\
    --model-name DeepSeek-Coder-V2-Lite-Instruct  \\
    --gpu-num 4 \\
    --dtype bf16

  1. 72 ~ 100 行可配置测试并发数以及输入输出组合！
  2. 自动监控 GPU 利用率并合并至结果 JSON 中。
EOF
}

# ---- 必填参数 ----
MODEL_PATH=""
MODEL_NAME=""

# ---- 默认参数 ----
HOST="localhost"
PORT=8000
DATASET_NAME="random"
CONFIG_FILE="config/vllm_bench_config.txt"

# ---- 参数解析 ----
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift ;;
        --model-name) MODEL_NAME="$2"; shift ;;
        --gpu-num) GPU_NUM="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --host) HOST="$2"; shift ;;
        --config-file) CONFIG_FILE="$2"; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# ---- 校验必填项 ----
if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ 错误: 必须传入 --model-path 参数！"
    exit 1
fi

if [[ -z "$GPU_NUM" ]]; then
    echo "❌ 错误: 必须传入 --gpu-num 参数！"
    exit 1
fi

# 如果未指定 model-name，则自动取路径最后一级目录名
if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME=$(basename "${MODEL_PATH%/}")
    echo "ℹ️ 未指定 --model-name，自动使用模型名: $MODEL_NAME"
fi

# ---- 定义测试组合 ----

# ---- 创建日志目录 ----
# LOG_DIR="./vllm_bench_logs/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./vllm_bench_logs/${MODEL_NAME}_tp${GPU_NUM}_dtype${DTYPE}_$(date +%Y%m%d_%H%M%S)"
CLIENT_LOG_DIR="$LOG_DIR/client_log"
mkdir -p "$CLIENT_LOG_DIR"

echo "============================================"
echo "🚀 启动 vLLM 批量性能测试"
echo "Model Path:  $MODEL_PATH"
echo "Model Name:  $MODEL_NAME"
echo "Host:        $HOST"
echo "Port:        $PORT"
echo "日志输出目录: $LOG_DIR"
echo "============================================"

# ---- 读取配置文件 ----
while read -r line || [[ -n "$line" ]]; do
    # 跳过注释或空行
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    # 读取并发数、输入长度、输出长度
    IFS=' ' read -r CONC INPUT_LEN OUTPUT_LEN <<< "$line"

    echo "▶️请求数: ${CONC}, 并发: ${CONC}, 输入: ${INPUT_LEN}, 输出: ${OUTPUT_LEN}"
    LOG_FILE=$LOG_DIR/${MODEL_NAME}_vllm_result.json
    GPU_LOG_DIR="${LOG_DIR}/gpu_utilization_c${CONC}_in${INPUT_LEN}_out${OUTPUT_LEN}"

    # GPU 监控启动
    python ../../gpu-monitor/mt-gmi-utilization.py \
      --gpu-num "${GPU_NUM}" \
      --interval 2 \
      --gpu-utilization-threshold 10.0 \
      --log-path "$GPU_LOG_DIR" \
      --metadata "model_name=${MODEL_NAME} concurrency=${CONC} input_len=${INPUT_LEN} output_len=${OUTPUT_LEN}" &

    GPU_MONITOR_PID=$!
    echo "⚙ GPU 监控启动, PID=$GPU_MONITOR_PID"
    
    # 配置 vLLM bench log 参数
    VLLM_BENCH_LOG_ARGS="--save-result \
            --append-result \
            --result-filename ${LOG_FILE} \
            --metadata model_name=${MODEL_NAME} concurrency=${CONC} input_len=${INPUT_LEN} output_len=${OUTPUT_LEN}"
            
    bash vllm_bench.sh \
      --model-path "$MODEL_PATH" \
      --model-name "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --max-concurrency "$CONC" \
      --num-prompts "$CONC" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --dataset "$DATASET_NAME" \
      --extra $VLLM_BENCH_LOG_ARGS >> ${CLIENT_LOG_DIR}/c"$CONC"_i"$INPUT_LEN"_o"$OUTPUT_LEN".log 2>&1

    # 终止 GPU 监控
    kill "$GPU_MONITOR_PID"
    wait "$GPU_MONITOR_PID" 2>/dev/null

    python merge_gpu_to_json.py \
      --json "$LOG_FILE" \
      --gpu-log ${GPU_LOG_DIR}/result.log 
    echo "added gpu utilization to ${LOG_FILE}"


    echo "✅ 已完成: 并发=${CONC}, 输入=${INPUT_LEN}, 输出=${OUTPUT_LEN}"
    echo "   日志: $LOG_FILE"
    echo "--------------------------------------------"

    echo "⏳ 等待系统稳定(60s)..."
    sleep 60  # 等待一段时间，确保系统稳定

done < "$CONFIG_FILE"

echo "🎯 所有批量测试已完成！结果日志保存在: $LOG_DIR"

