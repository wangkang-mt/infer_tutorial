#!/bin/bash

#########################################
# vLLM Serve 启动脚本
# 可传参：
#   --model-path
#   --served-model-name
#   --dtype
#   --max-model-len
#   --tp / --tensor-parallel
#   --pp / --pipeline-parallel
#   --gpu-mem
#
# 用法示例：
# bash start_vllm.sh \
#   --model-path /data/models/DeepSeek-R1-Distill-Qwen-32B \
#   --served-model-name DeepSeek-R1-Distill-Qwen-32B \
#   --dtype bfloat16 \
#   --tp 8 \
#   --max-model-len 32768 \
#   --gpu-mem 0.7
#########################################

show_help() {
cat << EOF
用法：
  bash $0 [参数...]

参数列表：
  --model-path PATH            （必填）模型路径
  --served-model-name NAME     服务模型名（默认自动从路径推断）
  --dtype DTYPE                默认: bfloat16
  --quantization TYPE          权重量化精度
  --max-model-len N            最大上下文长度
  --tp, --tensor-parallel N    默认: 8
  --pp, --pipeline-parallel N  默认: 1(不可更改)
  --gpu-mem FLOAT              GPU 内存利用率(默认 0.7)
  --host HOST                  默认 0.0.0.0
  --port PORT                  默认 8000
  --save-logs LOG_PATH             日志保存路径（默认保存当前目录）

示例：
  bash $0 \\
    --model-path /data/models/DeepSeek-R1-Distill-Qwen-32B \\
    --dtype bfloat16 \\
    --tp 8 \\
    --max-model-len 32768

EOF
}

# 默认参数
MODEL_PATH=""
MODEL_NAME=""
DTYPE="bfloat16"
TP=8
PP=1
GPU_MEM=0.7
PORT=8000
HOST="0.0.0.0"

# LOG 
LOG_FILE=""
LOG_DIR="./vllm_serve_logs"

EXTRA_ARGS=()

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
      -h|--help)
      show_help
      exit 0
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --served-model-name|--model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --quantization)
      TYPE="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_LEN="$2"
      shift 2
      ;;
    --tp|--tensor-parallel)
      TP="$2"
      shift 2
      ;;
    --pp|--pipeline-parallel)
      PP="$2"
      shift 2
      ;;
    --gpu-mem)
      GPU_MEM="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --save-logs)
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# 参数检查
if [[ -z "$MODEL_PATH" ]]; then
  echo "❌ 必须提供 --model-path"
  exit 1
fi

# 自动推断模型名
if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME=$(basename "$MODEL_PATH")
  # 去掉路径末尾的 "/"（如果有）
  MODEL_NAME=${MODEL_NAME%/}
  echo "⚠️ 未指定 --served-model-name, 自动推断为: $MODEL_NAME"
fi

if [[ -z "$LOG_FILE" ]]; then
    LOG_FILE="${LOG_DIR}/vllm_serve_${MODEL_NAME}_tp${TP}_pp${PP}_dtype${DTYPE}.log"
    mkdir -p "$LOG_DIR"
fi

echo "========================================"
echo "🚀 启动 vLLM 推理服务..."
echo " Model Path:         $MODEL_PATH"
echo " Served Model Name:  $MODEL_NAME"
echo " DType:              $DTYPE"
echo " Quantization:       $TYPE"
echo " Max Model Len:      ${MAX_LEN:-'(未设置)'}"
echo " TP:                 $TP"
echo " PP:                 $PP"
echo " GPU Mem Util:       $GPU_MEM"
echo " Host:               $HOST"
echo " Port:               $PORT"
echo "========================================"

# 启动命令
CMD=(
  vllm serve
  "$MODEL_PATH"
  --trust-remote-code
  --served-model-name "$MODEL_NAME"
  --gpu-memory-utilization "$GPU_MEM"
  --block-size 64
  --dtype "$DTYPE"
  --tensor-parallel-size "$TP"
  --pipeline-parallel-size "$PP"
  --host "$HOST"
  --port "$PORT"
  "${EXTRA_ARGS[@]}"
)


[[ -n "$MAX_LEN" ]] && CMD+=(--max-model-len "$MAX_LEN")
[[ -n "$TYPE" ]] && CMD+=(--quantization "$TYPE")

COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,10,12,14,16,18,20,24,28,30], "simple_cuda_graph": true}'
CMD+=(--compilation-config "$COMPILATION_CONFIG")

echo ""
echo "🚀 启动命令："
echo "VLLM_USE_V1=0 $(printf "%s " "${CMD[@]}")"
echo ""
echo ""

# 执行
export VLLM_USE_V1=0

echo "💾 日志: $LOG_FILE"
exec "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
