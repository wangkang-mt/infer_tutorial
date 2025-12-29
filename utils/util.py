import subprocess
from typing import Union
import importlib.metadata as importlib_metadata
from packaging.version import Version 



def get_package_info(package_name: str) -> str:
    try:
        version = importlib_metadata.version(package_name)
        return Version(version).base_version
    except importlib_metadata.PackageNotFoundError:
        return "unknown"



def run_cmd(command: Union[str, list[str]], shell: bool = True) -> tuple[int, str, str]:
    process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=shell,
        )
    
    return process.stdout.decode('utf-8'), process.stderr.decode('utf-8'), process.returncode


if __name__ == "__main__":
    print(get_package_info("vllm"))
    print(get_package_info("vllm_musa"))