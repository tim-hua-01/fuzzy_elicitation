"""Serve OLMo-3-32B-Think-DPO using vLLM's OpenAI-compatible API server."""

import subprocess
import sys

def main():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "allenai/Olmo-3-32B-Think-DPO",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "4096",
        "--trust-remote-code",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
