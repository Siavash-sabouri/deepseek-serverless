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
        "prompt": (
            "Create a professional Python script for educational cybersecurity purposes. "
            "The script must monitor a Linux system log file in real time, detect repeated "
            "failed SSH login attempts, count attempts per IP address, and print an alert "
            "when an IP exceeds a configurable threshold. "
            "Use clean structure, functions, comments, and follow best practices. "
            "Do not use external libraries. "
            "Minimum length: 40 lines of code."
        )
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
        print("Answer:\n")
        print(s["output"]["output"])
        break
    else:
        print("Status:", s["status"])
