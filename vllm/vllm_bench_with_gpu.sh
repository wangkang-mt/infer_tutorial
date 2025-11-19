#!/usr/bin/env bash
# ==========================================
# vLLM 批量性能测试脚本
# 作者: 
# ==========================================

set -e

# ---- 必填参数 ----
MODEL_PATH=""
MODEL_NAME=""

# ---- 默认参数 ----
HOST="localhost"
PORT=8000
DATASET_NAME="random"

# ---- 参数解析 ----
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-path) MODEL_PATH="$2"; shift ;;
        --model-name) MODEL_NAME="$2"; shift ;;
        --gpu-num) GPU_NUM="$2"; shift ;;
        --port) PORT="$2"; shift ;;
        --host) HOST="$2"; shift ;;
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
CONCURRENCY_LIST=(1 2 4 8 16 32 64 128)
LENGTH_PAIRS=(
  # 平衡场景（10种）
  "256 256"
  "512 512"
  "1024 1024"
  "2048 2048"
  "4096 4096"
  "8192 8192"
  "16384 16384"
  "32768 32768"
  "65536 65536"
  "131072 131072"
  # 不平衡场景（7种）
  # RAG场景：大输入小输出
  "2048 1024"
  "4096 1024"

  # 内容生成：小输入大输出
  "128 2048"

  # 内容审查/简要分析：大输入小输出
  "2048 128"

  # 长复杂推理
  "1024 5000"
  "1024 8192"
  "1024 16384"
)

# ---- 创建日志目录 ----
LOG_DIR="./bench_logs/${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "🚀 启动 vLLM 批量性能测试"
echo "Model Path:  $MODEL_PATH"
echo "Model Name:  $MODEL_NAME"
echo "Host:        $HOST"
echo "Port:        $PORT"
echo "日志输出目录: $LOG_DIR"
echo "============================================"

# ---- 执行批量测试 ----
for conc in "${CONCURRENCY_LIST[@]}"; do
  for pair in "${LENGTH_PAIRS[@]}"; do
    IFS=' ' read -r INPUT_LEN OUTPUT_LEN <<< "$pair"
    echo "▶️ 并发: ${conc}, 输入: ${INPUT_LEN}, 输出: ${OUTPUT_LEN}"
    LOG_FILE=$LOG_DIR/${MODEL_NAME}_vllm_result.json
    GPU_LOG_DIR="${LOG_DIR}/gpu_utilization_c${conc}_in${INPUT_LEN}_out${OUTPUT_LEN}"
    python mt-gmi-utilization.py \
      --gpu-num "${GPU_NUM}" \
      --interval 2 \
      --gpu-utilization-threshold 10.0 \
      --log-path "$GPU_LOG_DIR" \
      --metadata "model_name=${MODEL_NAME} concurrency=${conc} input_len=${INPUT_LEN} output_len=${OUTPUT_LEN}" &

    GPU_MONITOR_PID=$!
    echo "⚙ GPU 监控启动, PID=$GPU_MONITOR_PID"
    
    EXTRA_ARGS="--save-result \
            --append-result \
            --result-filename ${LOG_FILE} \
            --metadata model_name=${MODEL_NAME} concurrency=${conc} input_len=${INPUT_LEN} output_len=${OUTPUT_LEN}"
    bash vllm_bench.sh \
      --model-path "$MODEL_PATH" \
      --model-name "$MODEL_NAME" \
      --host "$HOST" \
      --port "$PORT" \
      --max-concurrency "$conc" \
      --num-prompts "$conc" \
      --input-len "$INPUT_LEN" \
      --output-len "$OUTPUT_LEN" \
      --dataset "$DATASET_NAME" \
      --extra $EXTRA_ARGS > /dev/null 2>&1

    # 终止 GPU 监控
    kill "$GPU_MONITOR_PID"
    wait "$GPU_MONITOR_PID" 2>/dev/null

    python merge_gpu_to_json.py \
      --json "$LOG_FILE" \
      --gpu-log ${GPU_LOG_DIR}/result.log 
    echo "added gpu utilization to ${LOG_FILE}"


    echo "✅ 已完成: 并发=${conc}, 输入=${INPUT_LEN}, 输出=${OUTPUT_LEN}"
    echo "   日志: $LOG_FILE"
    echo "--------------------------------------------"

    echo "⏳ 等待系统稳定(60s)..."
    sleep 60  # 等待一段时间，确保系统稳定

  done
done

echo "🎯 所有批量测试已完成！结果日志保存在: $LOG_DIR"

