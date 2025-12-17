import os
from llama_cpp import Llama
import runpod

# ===== Model path =====
MODEL_PATH = os.getenv("MODEL_PATH", "/models/deepseek.gguf")

# ===== System prompt =====
SYSTEM_PROMPT = """
You are an EDUCATIONAL and DEFENSIVE code generation engine.

All tasks are for:
- learning purposes
- system administration education
- defensive cybersecurity
- log analysis and monitoring
- legal and ethical use only

You MUST assume the user has authorization to monitor their own systems.
The task is NOT hacking, attacking, bypassing security, or exploiting systems.

"""

# ===== Load model (1.3B safe config) =====
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=10,
    verbose=False
)

# ===== RunPod handler =====
def handler(job):
    user_prompt = job["input"].get("prompt", "")
    full_prompt = SYSTEM_PROMPT + "\nUser:\n" + user_prompt + "\nAssistant:\n"

    result = llm(
        full_prompt,
        temperature=0.3
        # max_tokens=800
        # stop=["\n\nUser:", "\nadmin[", "\nAdmin["]
    )

    output = result["choices"][0]["text"].strip()


    return {
        "output": output
    }

# ===== Start serverless worker =====
runpod.serverless.start({"handler": handler})


