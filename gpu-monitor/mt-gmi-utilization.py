import subprocess
import os
import re
import psutil
import argparse
import time
import signal
import sys
import numpy as np


records = []

def signal_handler(sig, frame):
    if records:
        save_log(records, args.log_path, args.metadata, args)
    else:
        print("没有记录可保存")
    sys.exit(0)


# 注册信号处理
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill



def parse_metadata(kv_list):
    """Convert ['a=b', 'c=d'] into dict."""
    metadata = {}

    if not kv_list:
        return metadata

    parts = []
    for item in kv_list:
        parts.extend(item.split())

    for p in parts:
        if "=" not in p:
            raise ValueError(f"Metadata item '{p}' must be key=value")
        key, value = p.split("=", 1)
        metadata[key] = value
    return metadata


def parse_arguments():
    parser = argparse.ArgumentParser(description="Get GPU and CPU utilization.")

    # GPU monitoring parameters
    parser.add_argument("--gpu-num", type=int, default=1, help="Number of GPUs to monitor")

    # logging parameters
    parser.add_argument("--log-path", type=str, default="./logs/gpu_utilization", help="Path to log file")
    parser.add_argument("--metadata", nargs="*", help="Key-value metadata, e.g. --metadata model=qwen batch=16 gpu=A100")

    # monitoring parameters
    parser.add_argument("--interval", type=int, default=2, help="Interval in seconds between monitoring")
    parser.add_argument("--duration", type=int, default=0, help="Total duration in seconds for monitoring")

    parser.add_argument("--gpu-memory-threshold", type=float, default=0.0, help="Monitor only if used_memory >= this threshold (MiB)")
    parser.add_argument("--gpu-memory-delta", type=float, default=0.0, help="Monitor if used_memory increases by this delta + threshold (MiB) ")
    parser.add_argument("--gpu-utilization-threshold", type=float, default=0.0, help="Monitor only if gpu_utilization >= this threshold (%)")

    args = parser.parse_args()
    args.metadata = parse_metadata(args.metadata)
    return args


def average_gpu_status(gpu_status_list, args):
    selected_gpus = gpu_status_list[: args.gpu_num]
    total_gpu = 0
    total_temp = 0
    total_totalmem = 0
    total_usedmem = 0

    for (_, gpu_usage, temp, total_mem, used_mem) in selected_gpus:
        total_gpu += int(gpu_usage)
        total_temp += int(temp)
        total_totalmem += int(total_mem)
        total_usedmem += int(used_mem)

    avg_result = {
        "gpu_nums": args.gpu_num,
        "gpu_usage_avg": total_gpu / args.gpu_num,
        "temperature_avg": total_temp / args.gpu_num,
        "total_memory_avg": total_totalmem / args.gpu_num,
        "used_memory_avg": total_usedmem / args.gpu_num,
    }

    assert avg_result["total_memory_avg"] != selected_gpus[0][3], "Inconsistent total memory across GPUs"
    return avg_result



def get_gpu_utilization(args):
    """Get GPU utilization using mthreads-gmi command"""
    # Run the command to get GPU utilization info
    command = ["mthreads-gmi", "-q", "-d", "UTILIZATION,TEMPERATURE,MEMORY"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Command execution failed:", result.stderr)
        return

    # Get the output from the command
    output = result.stdout
    gpu_utilization = {}

    pattern = (
    r"(GPU\d+).*?"                              # GPU ID
    r"Gpu\s*:\s*(\d+)%.*?"                      # GPU utilization %
    r"GPU Current Temp\s*:\s*(\d+)C.*?"         # Temperature
    r"FB Memory Usage.*?"                       # Enter FB Memory section
    r"Total\s*:\s*(\d+)MiB.*?"                  # Total FB mem
    r"Used\s*:\s*(\d+)MiB"                      # Used FB mem
    )
    matches = re.findall(pattern, output, re.DOTALL)

    for match in matches:
        # time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        gpu_id = match[0]
        gpu_usage = match[1]
        temperature = match[2]
        memory_total = match[3]
        memory_usage = match[4]
        
        gpu_utilization[gpu_id] = (gpu_usage, memory_usage)

    # Print the results for GPU
    # for gpu_id, (gpu_usage, memory_usage) in gpu_utilization.items():
    #     print(f"{gpu_id}  - GPU: {gpu_usage}%, Mem: {memory_usage}MiB, Total Memory:{memory_total}MiB, 🌡: {temperature}C")



    avg_data = average_gpu_status(matches, args)
    return avg_data
    # save_log(avg_data, args.log_path, args.metadata)


def continuous_gpu_monitoring(args):
    """Loop monitoring"""

    global records
    start_time = time.time()
    if args.duration:

        while (time.time() - start_time) < args.duration:

            records.append(get_gpu_utilization(args))
            time.sleep(args.interval)
    else:
        while True:
            records.append(get_gpu_utilization(args))

            time.sleep(args.interval)

    
    # save_log(records, args.log_path, args.metadata)
       
    save_log(records, args.log_path, args.metadata, args)


    
def filter_records_with_threshold(records, memeory_key, memory_threshold, memory_delta, utilization_key, utilization_threshold, utilization_delta=0.0):

    effective_memory_threshold = memory_threshold + memory_delta
    effective_utilization_threshold = utilization_threshold + utilization_delta
    
    filetered_records = [
        r for r in records
        if r.get(memeory_key, 0) >= effective_memory_threshold and r.get(utilization_key, 0.0) >= effective_utilization_threshold
    ]
    return filetered_records


def filter_records_with_std(records, filter_field, std_factor=2.0):
    """Filter records based on standard deviation of utilization."""

    if len(records) < 3:
        return records

    records_array = np.array([r[filter_field] for r in records])
    mean = np.mean(records_array)
    std = np.std(records_array)
    low, high = mean - std_factor * std, mean + std_factor * std
    return [r for r in records if low <= r[filter_field] <= high]


def save_log(records, log_dir, metadata, args):

    os.makedirs(log_dir, exist_ok=True)

    # --- define raw log path ---
    raw_log_path = os.path.join(log_dir, "raw.log")
    # --- write raw data log ---
    with open(raw_log_path, "a") as raw_file:
        for record in records:
            raw_entry = {**metadata, **record}
            raw_file.write(str(raw_entry) + "\n")

    # filter records based on threshold and std
    if args.gpu_memory_threshold  or args.gpu_utilization_threshold:
        records = filter_records_with_threshold(
            records,
            memory_threshold=args.gpu_memory_threshold,
            memory_delta=args.gpu_memory_delta,
            memeory_key="used_memory_avg",
            utilization_key="gpu_usage_avg",
            utilization_threshold=args.gpu_utilization_threshold,
        )
    records = filter_records_with_std(records, filter_field="gpu_usage_avg", std_factor=2.0)


    # save average log
    result_log_path = os.path.join(log_dir, "result.log")
    keys = records[0].keys()
    length_data = len(records)
    avg_data = {key: sum(d[key] for d in records) / length_data for key in keys}

    with open(result_log_path, "a") as log_file:
        log_entry = {**metadata, **avg_data}
        log_file.write(str(log_entry) + "\n")





def get_cpu_utilization():
    # Get CPU utilization
    cpu_usage = psutil.cpu_percent(interval=1)
    # Get memory usage
    memory = psutil.virtual_memory()
    memory_usage = memory.percent

    # Print the results for CPU
    print(f"CPU - Usage: {cpu_usage}% Mem: {memory_usage}%")



if __name__ == "__main__":
    args = parse_arguments()
    continuous_gpu_monitoring(args)
    # get_gpu_utilization(args)
    # get_cpu_utilization()


