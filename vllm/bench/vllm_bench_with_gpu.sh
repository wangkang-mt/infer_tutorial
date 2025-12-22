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
  --model-name NAME         (可选) 模型服务名，缺省则自动从路径推断（不做大小写转换）
  --dtype xxx               (可选) 推理精度，仅用于生成日志标记
  --port PORT               (可选) 默认: 8000
  --host HOST               (可选) 默认: localhost
  --threshold THRESHOLD     (可选) 监控阈值，格式: "ttft:100 tpot:50", 单位 ms, 允许的指标: ttft, tpot, e2el
  --help, -h                显示此帮助信息
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

cleanup_threshold_monitor() {
  echo ""
  echo "🧹 正在清理后台监控进程..."

  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    echo "🛑 停止监控进程 PID=$MONITOR_PID"
    kill "$MONITOR_PID"
    wait "$MONITOR_PID" 2>/dev/null
  fi

  echo "✅ 清理完成"
}
trap cleanup_threshold_monitor EXIT INT TERM

auto_select_with_threshold() {
    # ---- 启动实时监控程序（后台）----
    if [[ -n "$THRESHOLD" ]]; then
      echo "👀 启动实时监控程序，阈值: $THRESHOLD"
      python ./utils/auto_batch_selector.py \
        --json-file "$LOG_FILE" \
        --threshold "$THRESHOLD" \
        --output "$best_signal_file" &
      MONITOR_PID=$!
      echo "📡 监控进程 PID=$MONITOR_PID"
    fi
}



parse_args() {
    # ---- 参数解析 ----
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            --model-path) MODEL_PATH="$2"; shift ;;
            --model-name) MODEL_NAME="$2"; shift ;;
            --gpu-num) GPU_NUM="$2"; shift ;;
            --port) PORT="$2"; shift ;;
            --host) HOST="$2"; shift ;;
            --threshold) THRESHOLD="$2"; shift ;;
            --dtype) DTYPE="$2"; shift ;;
            --help|-h) show_help; exit 0 ;;
            *) echo "未知参数: $1"; exit 1 ;;
        esac
        shift
    done

    # 如果未指定 model-name，则自动取路径最后一级目录名
    if [[ -z "$MODEL_NAME" ]]; then
        MODEL_NAME=$(basename "${MODEL_PATH%/}")
        echo "ℹ️ 未指定 --model-name，自动使用模型名: $MODEL_NAME"
    fi

    # ---- 校验必填项 ----
    if [[ -z "$MODEL_PATH" ]]; then
        echo "❌ 错误: 必须传入 --model-path 参数！"
        exit 1
    fi
    if [[ -z "$GPU_NUM" ]]; then
        echo "❌ 错误: 必须传入 --gpu-num 参数！"
        exit 1
    fi
}
parse_args "$@"

# ---- 默认参数 ----
HOST="localhost"
PORT=8000
DATASET_NAME="random"

# ---- 创建日志目录 ----
LOG_DIR="./vllm_bench_logs/${MODEL_NAME}_tp${GPU_NUM}${DTYPE:+_dtype${DTYPE}}_$(date +%Y%m%d_%H%M%S)"
CLIENT_LOG_DIR="$LOG_DIR/client_log"
mkdir -p "$CLIENT_LOG_DIR"


## ---- 定义结果日志文件 ----
best_signal_file="./utils/best_signal.json"
LOG_FILE="$LOG_DIR/${MODEL_NAME}_vllm_result.json"

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


echo "============================================"
echo "🚀 启动 vLLM 批量性能测试"
echo "Model Path:  $MODEL_PATH"
echo "Model Name:  $MODEL_NAME"
echo "Host:        $HOST"
echo "Port:        $PORT"
echo "日志输出目录: $LOG_DIR"
echo "============================================"

# ---- 启动自动选择监控（如果设置了阈值）----
auto_select_with_threshold

# ---- 执行批量测试 ----
for pair in "${LENGTH_PAIRS[@]}"; do
  for conc in "${CONCURRENCY_LIST[@]}"; do
    IFS=' ' read -r INPUT_LEN OUTPUT_LEN <<< "$pair"
    echo "▶️ 并发: ${conc}, 输入: ${INPUT_LEN}, 输出: ${OUTPUT_LEN}"

    GPU_LOG_DIR="${LOG_DIR}/gpu_utilization_c${conc}_in${INPUT_LEN}_out${OUTPUT_LEN}"
    python ../../gpu-monitor/mt-gmi-utilization.py \
      --gpu-num "${GPU_NUM}" \
      --interval 2 \
      --gpu-utilization-threshold 10.0 \
      --log-path "$GPU_LOG_DIR" \
      --metadata "model_name=${MODEL_NAME} concurrency=${conc} input_len=${INPUT_LEN} output_len=${OUTPUT_LEN}" &

    GPU_MONITOR_PID=$!
    echo "⚙ GPU 监控启动, PID=$GPU_MONITOR_PID"
    
    VLLM_BENCH_LOG_ARGS="--save-result \
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
      --extra $VLLM_BENCH_LOG_ARGS >> ${CLIENT_LOG_DIR}/c"$conc"_i"$INPUT_LEN"_o"$OUTPUT_LEN".log 2>&1

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

    if [[ -f "$best_signal_file" ]]; then
      echo "⚠️ input_len=${INPUT_LEN}, output_len=${OUTPUT_LEN} 在满足${THRESHOLD} 条件下的最优并发数为：${conc}"
      rm "$best_signal_file"
    fi

  done
done

echo "🎯 所有批量测试已完成！结果日志保存在: $LOG_DIR"

