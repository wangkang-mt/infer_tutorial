import json
import time
import argparse
import signal
import sys
import os
from pathlib import Path
from subprocess import run
from typing import Dict, Set, Tuple, Any, Optional, Generator, List
from dataclasses import dataclass
from collections import OrderedDict
import threading


@dataclass
class TestRecord:
    """测试记录数据结构"""
    input_len: int
    output_len: int
    concurrency: int
    mean_ttft_ms: Optional[float] = None
    mean_tpot_ms: Optional[float] = None
    mean_e2el_ms: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TestRecord':
        """从字典创建TestRecord"""
        return cls(
            input_len=data.get("input_len", 0),
            output_len=data.get("output_len", 0),
            concurrency=data.get("concurrency", 0),
            mean_ttft_ms=data.get("mean_ttft_ms"),
            mean_tpot_ms=data.get("mean_tpot_ms"),
            mean_e2el_ms=data.get("mean_e2el_ms")
        )


class SignalWriter:
    """信号文件写入器，用于通知最佳配置"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lock = threading.Lock()
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        output_dir = os.path.dirname(self.filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def write_best(self, record: TestRecord):
        """写入最佳配置到信号文件"""
        payload = {
            "input_len": record.input_len,
            "output_len": record.output_len,
            "best_batch": record.concurrency,
            "timestamp": time.time()
        }
        with self.lock:
            with open(self.filepath, "w", encoding='utf-8') as f:
                json.dump(payload, f, indent=2)
            print(f"📢 已写入信号文件: {self.filepath}")


class JsonTailReader:
    """高效读取追加的JSON日志文件"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.position = 0
        self._check_file()
    
    def _check_file(self):
        """检查文件是否存在，不存在则等待"""
        while not os.path.exists(self.filepath):
            print(f"⚠️  等待文件出现: {self.filepath}")
            time.sleep(2)
    
    def read_new_lines(self) -> Generator[Dict, None, None]:
        """只读取新增的行"""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                # 如果文件被清空或缩小，重置位置
                current_size = os.path.getsize(self.filepath)
                if current_size < self.position:
                    print(f"📄 文件被截断或清空，重置读取位置")
                    self.position = 0
                
                f.seek(self.position)
                lines = f.readlines()
                self.position = f.tell()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        # 跳过不完整的JSON行
                        continue
        except (FileNotFoundError, IOError) as e:
            print(f"❌ 文件读取错误: {e}")
            time.sleep(2)
    
    def reset(self):
        """重置读取位置到文件开头"""
        self.position = 0


def parse_threshold(s: str) -> Dict[str, float]:
    """解析阈值字符串"""
    res = {}
    if not s.strip():
        return res
    
    for item in s.split():
        if ':' not in item:
            continue
        try:
            k, v = item.split(":", 1)
            res[k.strip()] = float(v.strip())
        except ValueError:
            print(f"⚠️  忽略无效的阈值项: {item}")
    return res


def satisfy(record: TestRecord, threshold: Dict[str, float]) -> bool:
    """检查记录是否满足阈值条件"""
    if not threshold:
        return True
    
    checks = {
        'ttft': 'mean_ttft_ms',
        'tpot': 'mean_tpot_ms', 
        'e2el': 'mean_e2el_ms'
    }
    
    for thr_key, rec_attr in checks.items():
        if thr_key in threshold:
            rec_value = getattr(record, rec_attr, None)
            if rec_value is not None and rec_value > threshold[thr_key]:
                return False
    return True


class ResultManager:
    """管理最佳结果和去重"""
    
    def __init__(self, max_seen_size: int = 10000):
        self.seen: Set[Tuple] = set()
        self.best: Dict[Tuple, TestRecord] = OrderedDict()
        self.max_seen_size = max_seen_size
        self.lock = threading.RLock()
    
    def add_record(self, record: TestRecord) -> bool:
        """添加记录，返回是否是新记录"""
        with self.lock:
            key = (record.input_len, record.output_len, record.concurrency)
            if key in self.seen:
                return False
            
            self.seen.add(key)
            self._cleanup_seen()
            return True
    
    def _cleanup_seen(self):
        """清理seen集合，防止内存泄漏"""
        if len(self.seen) > self.max_seen_size:
            # 转换为列表，移除最旧的一半
            seen_list = list(self.seen)
            self.seen = set(seen_list[len(seen_list)//2:])
            print(f"🧹 清理seen集合，剩余 {len(self.seen)} 条记录")
    
    def update_best(self, record: TestRecord, threshold: Dict[str, float]) -> Tuple[bool, Optional[TestRecord]]:
        """更新最佳结果，返回(是否更新, 旧的最佳结果)"""
        with self.lock:
            io_key = (record.input_len, record.output_len)
            
            if satisfy(record, threshold):
                old_best = self.best.get(io_key)
                if old_best is None or record.concurrency > old_best.concurrency:
                    self.best[io_key] = record
                    return True, old_best
            return False, None
    
    def get_best(self, io_key: Tuple) -> Optional[TestRecord]:
        """获取指定IO键的最佳结果"""
        with self.lock:
            return self.best.get(io_key)
    
    def get_all_bests(self) -> List[TestRecord]:
        """获取所有最佳结果"""
        with self.lock:
            return list(self.best.values())
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            return {
                "seen_count": len(self.seen),
                "best_count": len(self.best),
                "io_configs": list(self.best.keys())
            }


class OutputWriter:
    """线程安全的输出写入器，为每个IO配置只保留最新的最佳记录"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lock = threading.Lock()
        self._ensure_output_dir()
        
        # 内存缓存：为每个IO配置存储最佳记录
        # key: (input_len, output_len)
        # value: 最佳TestRecord
        self.best_records: Dict[Tuple, TestRecord] = {}
        
        # 加载已有的最佳记录（如果文件已存在）
        self._load_existing_bests()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        output_dir = os.path.dirname(self.filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def _load_existing_bests(self):
        """从现有文件中加载最佳记录"""
        if not os.path.exists(self.filepath):
            return
        
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        record = TestRecord.from_dict(data)
                        io_key = (record.input_len, record.output_len)
                        
                        # 保留并发数最大的记录
                        if io_key in self.best_records:
                            if record.concurrency > self.best_records[io_key].concurrency:
                                self.best_records[io_key] = record
                        else:
                            self.best_records[io_key] = record
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            print(f"⚠️  加载现有最佳记录时出错: {e}")
    
    def write_record(self, record: TestRecord):
        """写入或更新最佳记录"""
        with self.lock:
            io_key = (record.input_len, record.output_len)
            
            # 检查是否需要更新
            need_update = False
            if io_key not in self.best_records:
                need_update = True
                self.best_records[io_key] = record
            elif record.concurrency > self.best_records[io_key].concurrency:
                need_update = True
                self.best_records[io_key] = record
            
            # 如果需要更新，重写整个文件
            if need_update:
                self._rewrite_file()
    
    def _rewrite_file(self):
        """重写整个最佳记录文件"""
        try:
            # 先写入临时文件，然后原子替换
            temp_file = self.filepath + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                for record in self.best_records.values():
                    f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
            
            # 原子替换文件
            os.replace(temp_file, self.filepath)
            
        except Exception as e:
            print(f"❌ 写入最佳记录文件时出错: {e}")
            # 如果出错，尝试直接写入
            try:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    for record in self.best_records.values():
                        f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
            except Exception as e2:
                print(f"❌ 直接写入也失败: {e2}")
    
    def get_best_for_io(self, input_len: int, output_len: int) -> Optional[TestRecord]:
        """获取指定IO配置的最佳记录"""
        with self.lock:
            return self.best_records.get((input_len, output_len))
    
    def get_all_bests(self) -> List[TestRecord]:
        """获取所有最佳记录"""
        with self.lock:
            return list(self.best_records.values())
    
    def flush(self):
        """强制刷新缓冲区"""
        pass


class MonitorMode:
    """监控模式"""
    
    def __init__(self, args, threshold: Dict[str, float]):
        self.args = args
        self.threshold = threshold
        self.reader = JsonTailReader(args.log_file)
        self.result_manager = ResultManager()
        self.writer = OutputWriter(args.output)
        self.running = False
        self.signal_writer = SignalWriter(args.signal_file)
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """信号处理函数"""
        print(f"\n🛑 接收到信号 {sig}，正在优雅退出...")
        self.running = False
    
    def _process_record(self, record_data: Dict) -> bool:
        """处理单条记录"""
        try:
            record = TestRecord.from_dict(record_data)
            
            # 检查是否为重复记录
            if not self.result_manager.add_record(record):
                return False
            
            io_key = (record.input_len, record.output_len)
            updated, old_best = self.result_manager.update_best(record, self.threshold)
            
            if updated:
                # 写入最佳记录（OutputWriter会自动去重，只保留最大的并发数）
                self.writer.write_record(record)
                
                old_batch = old_best.concurrency if old_best else "无"
                print(f"✅ IO={io_key} batch={record.concurrency} 满足 (之前: {old_batch})")
                return True
            else:
                current_best = self.result_manager.get_best(io_key)
                if current_best:
                    # 只有当发现不满足条件的更大batch时，才确认最佳
                    if current_best.concurrency < record.concurrency:
                        print(f"🎯 发现最佳配置！IO={io_key} 最佳batch={current_best.concurrency}")
                        self.signal_writer.write_best(current_best)
                    
                    print(f"⛔ IO={io_key} batch={record.concurrency} 不满足，保留 batch={current_best.concurrency}")
                else:
                    print(f"⚠️  IO={io_key} batch={record.concurrency} 不满足，尚无最佳")
                return False
                
        except Exception as e:
            print(f"❌ 处理记录时出错: {e}")
            return False
    
    def run(self):
        """运行监控循环"""
        print("👀 开始实时监控模式")
        print(f"📄 监控文件: {self.args.log_file}")
        print(f"💾 输出文件: {self.args.output} (每个IO配置只保留最佳记录)")
        print(f"📢 信号文件: {self.args.signal_file}")
        print(f"📊 阈值配置: {self.threshold}")
        print("按下 Ctrl+C 停止监控\n")
        
        self.running = True
        empty_cycles = 0
        
        while self.running:
            processed = 0
            for record_data in self.reader.read_new_lines():
                self._process_record(record_data)
                processed += 1
            
            # 动态调整休眠时间
            if processed > 0:
                empty_cycles = 0
                time.sleep(1)
            else:
                empty_cycles += 1
                sleep_time = min(2 + empty_cycles, 10)
                time.sleep(sleep_time)
            
            # 每10次循环打印一次统计信息
            if empty_cycles % 10 == 0:
                stats = self.result_manager.get_stats()
                print(f"📈 统计: 已处理 {stats['seen_count']} 条，最佳配置 {stats['best_count']} 个")
        
        print("👋 监控模式已停止")
        self._print_final_stats()
    
    def _print_final_stats(self):
        """打印最终统计信息"""
        stats = self.result_manager.get_stats()
        print("\n📊 最终统计:")
        print(f"  总处理记录数: {stats['seen_count']}")
        print(f"  最佳配置数量: {stats['best_count']}")
        print(f"  输出文件: {self.args.output}")
        
        if stats['best_count'] > 0:
            print("\n🎯 最佳配置汇总:")
            for io_key in stats['io_configs']:
                best = self.result_manager.get_best(io_key)
                if best:
                    print(f"  IO={io_key}: batch={best.concurrency}")


class BinarySearchMode:
    """二分搜索模式"""
    
    def __init__(self, args, threshold: Dict[str, float]):
        self.args = args
        self.threshold = threshold
        self.writer = OutputWriter(args.output)
        self.reader = JsonTailReader(args.log_file)
        
        # 验证必要参数
        if not args.bench_cmd_template:
            raise ValueError("二分搜索模式需要 --bench-cmd-template 参数")
    
    def run_batch(self, batch: int):
        """运行指定批次的测试"""
        cmd = self.args.bench_cmd_template.format(batch=batch)
        print(f"▶️  执行测试: {cmd}")
        
        try:
            result = run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️  命令执行失败: {result.stderr}")
        except Exception as e:
            print(f"❌ 执行命令时出错: {e}")
    
    def get_latest_result(self) -> Optional[TestRecord]:
        """获取最新的测试结果"""
        try:
            # 等待新结果出现
            time.sleep(1)
            
            # 读取最后一行
            last_data = None
            for data in self.reader.read_new_lines():
                last_data = data
            
            if last_data:
                return TestRecord.from_dict(last_data)
            else:
                print("⚠️  未找到测试结果")
                return None
        except Exception as e:
            print(f"❌ 获取测试结果时出错: {e}")
            return None
    
    def run(self):
        """执行二分搜索"""
        print("🎯 开始二分搜索模式")
        print(f"📄 日志文件: {self.args.log_file}")
        print(f"💾 输出文件: {self.args.output} (每个IO配置只保留最佳记录)")
        print(f"📊 阈值配置: {self.threshold}")
        print(f"🔍 搜索范围: [{self.args.min_batch}, {self.args.max_batch}]\n")
        
        lo, hi = self.args.min_batch, self.args.max_batch
        best_record = None
        iteration = 0
        
        while lo <= hi:
            iteration += 1
            mid = (lo + hi) // 2
            print(f"\n📋 迭代 {iteration}: 测试 batch={mid} [范围: {lo}-{hi}]")
            
            # 重置读取位置，确保获取最新结果
            self.reader.reset()
            
            # 运行测试
            self.run_batch(mid)
            
            # 获取结果
            record = self.get_latest_result()
            if not record:
                print("⚠️  跳过本次测试，继续...")
                hi = mid - 1
                continue
            
            print(f"   结果: TTFT={record.mean_ttft_ms}ms, TPOT={record.mean_tpot_ms}ms")
            
            if satisfy(record, self.threshold):
                print(f"   ✅ 满足阈值，尝试更大的batch")
                best_record = record
                self.writer.write_record(record)
                lo = mid + 1
            else:
                print(f"   ❌ 不满足阈值，尝试更小的batch")
                hi = mid - 1
        
        print("\n🎯 二分搜索完成")
        if best_record:
            print(f"🏆 最佳配置: batch={best_record.concurrency}")
            print(f"   输入长度: {best_record.input_len}")
            print(f"   输出长度: {best_record.output_len}")
        else:
            print("⚠️  未找到满足阈值的配置")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="性能测试结果监控与优化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 监控模式
  python script.py --log-file results.jsonl --threshold "ttft:100 tpot:50" --output best.jsonl
  
  # 二分搜索模式  
  python script.py --mode binary --log-file results.jsonl --threshold "ttft:100" \\
                   --output best.jsonl --bench-cmd-template "python bench.py --batch {batch}"
        """
    )
    
    parser.add_argument("--log-file", required=True,
                       help="JSONL格式的日志文件路径")
    parser.add_argument("--threshold", required=True,
                       help="阈值配置，格式: 'ttft:100 tpot:50 e2el:500'")
    parser.add_argument("--output", required=True,
                       help="输出结果文件路径")
    parser.add_argument("--mode", choices=["monitor", "binary"], default="monitor",
                       help="运行模式: monitor(监控) 或 binary(二分搜索)")
    parser.add_argument("--bench-cmd-template",
                       help="二分搜索模式使用的基准测试命令模板，使用 {batch} 占位符")
    parser.add_argument("--min-batch", type=int, default=1,
                       help="二分搜索的最小批次大小")
    parser.add_argument("--max-batch", type=int, default=128,
                       help="二分搜索的最大批次大小")
    parser.add_argument("--max-seen-size", type=int, default=10000,
                       help="监控模式的最大去重集合大小")
    parser.add_argument("--signal-file", default="best_signal.json",
                       help="当发现最佳 batch 时写入的信号文件")
    
    args = parser.parse_args()
    
    try:
        # 解析阈值
        threshold = parse_threshold(args.threshold)
        if not threshold:
            print("⚠️  警告: 阈值配置为空，将接受所有结果")
        
        # 创建输出目录
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        
        # 根据模式运行
        if args.mode == "monitor":
            monitor = MonitorMode(args, threshold)
            monitor.run()
        else:
            searcher = BinarySearchMode(args, threshold)
            searcher.run()
            
    except KeyboardInterrupt:
        print("\n👋 用户中断执行")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()