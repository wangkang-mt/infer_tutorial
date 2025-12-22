# 1. 起服务
（略）假设已通过vllm serve 启动 serve 服务：
- model: /data/model/Qwen3-32B
- served-model-name: Qwen3-32B
- tp: 8
- dtype: bfloat16


# 2. 单次 bench 测试
```shell
bash vllm_bench.sh \
    --model-path /data/model/Qwen3-32B \
    --model-name Qwen3-32B \
    --max-concurrency 32 \
    --input-len 1024 \
    --output-len 1024

# more
bash vllm_bench.sh -h
```

# 3. 批量测试
```shell

bash vllm_bench_with_gpu.sh \
    --model-path /data/model/Qwen3-32B \
    --dtype fp16 \
    --gpu-num 8
```
- gpu-num 和 dtype： 自动生成./bench_logs/Qwen3-32B_tp8_dtypefp16_< time > 目录；另外脚本自动检测每个测试组合对应GPU信息，如利用率，显存占用以及温度等信息，gpu-num 用于指定前 N 张卡用来监控
- vllm_bench_with_gpu.sh：72 ~ 100 行可配置测试并发数以及输入输出组合！

![vllm_bench](./assets/vllm_bench.png)

## 3.1 基于阈值的最佳性能探索
在给定 TTFT，TPOT 及 E2EL 延迟约束的前提下，自动探索不同输入 / 输出长度组合下，
**满足阈值条件的最大并发（batch size）**，以获得受约束条件下的最优性能。  
当某一固定 IO 组合下，已经找到满足阈值的最大 batch 后，
将自动跳过该 IO 组合的后续测试，进入下一组 IO 配置。
```shell
# 示例：在 TTFT ≤ 400ms 且 TPOT ≤ 50ms 的约束条件下进行测试：
# --threshold THRESHOLD     (可选) 监控阈值，格式: "ttft:100 tpot:50", 单位 ms, 允许的指标: ttft, tpot, e2el
bash vllm_bench_with_gpu.sh \
    --model-path /data/model/Qwen3-32B \
    --dtype fp16 \
    --gpu-num 8 \
    --threshold "ttft:400 tpot:50"
```  
## 3.2 log 解析
log 自动保存在./bench_logs/Qwen3-32B_tp8_dtypefp16_< time > 目录：
- Qwen3-32B_vllm_result.json 保存每个测试项的结果（已自动拼接 GPU util 等相关信息）,保存形式参看[result.json](realtime_bench_plot/test_log.json)
- gpu_utilization_c<并发数>_in<输入>_out<输出> ：保存每次 benchmark 测试对应的 GPU 监控信息，包括显存占用、GPU 利用率及温度。
  - raw.log: 记录时间序列的前 gpu-num 张 GPU 平均状态，包括显存占用、GPU 利用率和温度，默认每 2 秒采样一次。
  - result.log: 基于 raw.log 筛选并统计 GPU 数据，默认条件为 GPU 利用率 >10 且显存占用标准差 <2，输出时间维度的平均结果。
- client_log：保存没个测试项对应 vllm serve bench 原生日志

## 3.2 实时解析
可实时监控各测试项性能，用于探索性能。
```shell
pip install streamlit

streamlit run realtime_bench_plot/realtime_bench_plot.py -- \
    --json-file ./bench_logs/Qwen3-32B_tp8_dtypefp16_< time >/Qwen3-32B_vllm_result.json \
    --metadata tp=8 dtype=fp16

# --metedata 元信息，用于显示在 WebUI 中用于标识当前服务启动项等配置信息
```
![bench plot](./assets/bench_plot.png)