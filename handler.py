import os
import requests
from llama_cpp import Llama

MODEL_PATH = "./deepseek.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Model downloaded.")

# Load model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=35
)

# Run inference
def handler(event):
    prompt = event.get("input", "")
    output = llm(prompt)
    return {"output": output["choices"][0]["text"]}
