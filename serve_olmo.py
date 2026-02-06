"""Serve OLMo-3-32B-Think-DPO using vLLM's OpenAI-compatible API server."""

import subprocess
import sys
from pathlib import Path

MODEL_ID = "allenai/Olmo-3-32B-Think-DPO"
DOWNLOAD_DIR = str(Path(__file__).resolve().parent / "models")

def main():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--download-dir", DOWNLOAD_DIR,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--dtype", "bfloat16",
        "--max-model-len", "16384",
        "--trust-remote-code",
        "--served-model-name", "olmo-3-32b-think-dpo",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
