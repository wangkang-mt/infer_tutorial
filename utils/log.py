from dataclasses import dataclass, asdict
import sys
import json
import socket
import time
from datetime import datetime


@dataclass
class LogContext:
    run_id: str
    host: str
    pid: int

    gpu_arch: str
    gpu_count: int

    framework: str
    framework_version: str

    model_name: str
    model_path: str

    profile_name: str | None = None



class LogManager:
    def __init__(self, context: LogContext, stream=sys.stdout):
        self.context = context
        self.stream = stream

    def _emit(self, payload: dict):
        payload.update({
            "ts": datetime.utcnow().isoformat() + "Z",
            "run_id": self.context.run_id,
            "host": self.context.host,
            "pid": self.context.pid,
        })
        self.stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.stream.flush()

    def event(self, name: str, **fields):
        payload = {
            "type": "event",
            "event": name,
            **asdict(self.context),
            **fields,
        }
        self._emit(payload)


    def info(self, message: str, **fields):
        self._message("INFO", message, **fields)

    def warning(self, message: str, **fields):
        self._message("WARNING", message, **fields)

    def error(self, message: str, **fields):
        self._message("ERROR", message, **fields)

    def _message(self, level: str, message: str, **fields):
        payload = {
            "type": "message",
            "level": level,
            "message": message,
            **asdict(self.context),
            **fields,
        }
        self._emit(payload)
# 
# ctx = LogContext(
#     run_id="deepseek-1736491234-node01",
#     host=socket.gethostname(),
#     pid=os.getpid(),
#     gpu_arch="S4000",
#     gpu_count=8,
#     framework="vllm",
#     framework_version="0.4.2",
#     model_name="deepseek",
#     model_path="/models/deepseek"
# )

# log = LogManager(ctx)

# log.event("startup")

# log.event(
#     "env_detected",
#     gpu_arch=ctx.gpu_arch,
#     framework=ctx.framework,
#     framework_version=ctx.framework_version
# )

# log.event(
#     "profile_selected",
#     profile_name="s4000_vllm_mla"
# )

# log.event(
#     "params_resolved",
#     block_size=16,
#     tensor_parallel_size=8,
#     pipeline_parallel_size=2
# )

# log.info("Launching vLLM server")

# log.event(
#     "command_built",
#     command="vllm serve ..."
# )
