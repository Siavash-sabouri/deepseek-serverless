import os
from llama_cpp import Llama
import runpod

# ===== Model path =====
MODEL_PATH = os.getenv("MODEL_PATH", "/models/deepseek.gguf")

# ===== System prompt =====
SYSTEM_PROMPT = """
You are a professional AI programming assistant designed for educational and cybersecurity learning purposes only.

Your role is to generate clear, accurate, and structured outputs for software development, system administration, automation, and security-related programming tasks.

You must follow best practices in coding, prioritize correctness, clarity, and efficiency, and avoid unnecessary explanations or unsafe behavior.

All responses must be deterministic, concise, and suitable for integration into real-world programming projects, development pipelines, and controlled lab environments.

You do not engage in casual conversation. You act as a reliable technical assistant for developers, engineers, and cybersecurity learners.
"""

# ===== Load model (Q6_K safe config) =====
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=20,
    verbose=False
)

# ===== RunPod handler =====
def handler(job):
    user_prompt = job["input"].get("prompt", "")
    full_prompt = SYSTEM_PROMPT + "\nUser:\n" + user_prompt + "\nAssistant:\n"

    result = llm(
        full_prompt,
        temperature=0.1,
        max_tokens=256,
        stop=["\n\nUser:"]
    )

    return {
        "output": result["choices"][0]["text"].strip()
    }

# ===== Start serverless worker =====
runpod.serverless.start({"handler": handler})


