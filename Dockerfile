FROM python:3.10-slim

RUN apt update && apt install -y git curl && pip install --no-cache-dir llama-cpp-python runpod

COPY . /app
WORKDIR /app

CMD ["python3", "-m", "runpod"]
