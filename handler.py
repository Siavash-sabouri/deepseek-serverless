import os
import requests
from llama_cpp import Llama
import runpod  # Required for serverless

MODEL_PATH = "deepseek.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded.")

# Load model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=35
)

# Required format for RunPod
def handler(job):
    prompt = job["input"]["prompt"]
    result = llm(prompt)
    return {"output": result["choices"][0]["text"]}

# Start the worker
runpod.serverless.start({"handler": handler})
