import os
import requests
import time

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")

headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

payload = {
    "input": {
        "prompt": "You are a helpful assistant. Answer in one short sentence: What is DeepSeek?"
    }
}

# send job
r = requests.post(
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/run",
    headers=headers,
    json=payload
)

job_id = r.json()["id"]
print("Job ID:", job_id)

# wait for result
while True:
    time.sleep(5)
    s = requests.get(
        f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/status/{job_id}",
        headers=headers
    ).json()

    if s["status"] == "COMPLETED":
        print("Answer:")
        print(s["output"]["output"])
        break
    else:
        print("Status:", s["status"])
