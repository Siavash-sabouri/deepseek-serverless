import os
import requests
import time
import sys

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT")

if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT:
    print("ERROR: RUNPOD_API_KEY or RUNPOD_ENDPOINT not set")
    sys.exit(1)

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
            "Minimum length: 40 lines of code. "
            "Output ONLY the complete Python script."
        )
    }
}

# send job
response = requests.post(
    f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/run",
    headers=headers,
    json=payload,
    timeout=30
)

response.raise_for_status()
job_id = response.json()["id"]
print("Job ID:", job_id)

# wait for result
while True:
    time.sleep(5)

    status_response = requests.get(
        f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT}/status/{job_id}",
        headers=headers,
        timeout=30
    )
    status_response.raise_for_status()

    data = status_response.json()
    status = data["status"]
    print("Status:", status)

    if status == "COMPLETED":
        output = data.get("output", {}).get("output", "")

        if output.startswith("ERROR:"):
            print("Model returned incomplete code.")
            print(output)
        else:
            print("\n===== GENERATED CODE =====\n")
            print(output)

        break

    if status in ("FAILED", "CANCELLED"):
        print("Job failed.")
        break
