import os
import requests
from llama_cpp import Llama
import runpod  # Required for RunPod serverless

# Model file info
MODEL_PATH = "deepseek.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"


SYSTEM_PROMPT = """
You are a professional AI programming assistant designed for educational and cybersecurity learning purposes only.

Your role is to generate clear, accurate, and structured outputs for software development, system administration, automation, and security-related programming tasks.

You must follow best practices in coding, prioritize correctness, clarity, and efficiency, and avoid unnecessary explanations or unsafe behavior.

All responses must be deterministic, concise, and suitable for integration into real-world programming projects, development pipelines, and controlled lab environments.

You do not engage in casual conversation. You act as a reliable technical assistant for developers, engineers, and cybersecurity learners.

"""



# Download the model if it's not already downloaded
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded.")

# Load the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=35
)

# Required RunPod handler format

def handler(job):
    user_prompt = job["input"]["prompt"]
    full_prompt = SYSTEM_PROMPT + "\n" + user_prompt

    result = llm(
        full_prompt,
        max_tokens=512,
        temperature=0.3
    )

    return {"output": result["choices"][0]["text"]}

# Start the RunPod worker
runpod.serverless.start({"handler": handler})


