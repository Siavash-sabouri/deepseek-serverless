import os
from llama_cpp import Llama
import runpod

# ===== Model path =====
MODEL_PATH = os.getenv("MODEL_PATH", "/models/deepseek.gguf")

# ===== System prompt =====
SYSTEM_PROMPT = """
You are a CODE GENERATION ENGINE.

RULES (MANDATORY):
- You MUST output ONLY executable Python code.
- You MUST output a COMPLETE Python script.
- You MUST NOT include explanations, markdown, or text outside code.
- You MUST NOT role-play or simulate terminals or users.
- You MUST stop immediately after the script ends.

IMPORTANT:
Output only Python code.
No summaries.
No descriptions.
No markdown.
No extra text.

FAILURE CONDITIONS:
- Any text outside Python code is an error.
- Any partial or cut script is an error.
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
        temperature=0.3,
        max_tokens=800
#        stop=["\n\nUser:", "\nadmin[", "\nAdmin["]
    )

    return {
        "output": result["choices"][0]["text"].strip()
    }

# ===== Start serverless worker =====
runpod.serverless.start({"handler": handler})


