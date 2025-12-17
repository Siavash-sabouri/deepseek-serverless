FROM python:3.10

WORKDIR /app

RUN apt update && apt install -y \
    git \
    curl \
    build-essential \
    cmake \
    && pip install --no-cache-dir runpod llama-cpp-python

# download 1.3B Instruct model (faster startup)
RUN mkdir -p /models && \
    curl -L \
    https://huggingface.co/TheBloke/deepseek-coder-1.3B-instruct-GGUF/resolve/main/deepseek-coder-1.3b-instruct.Q4_K_M.gguf \
    -o /models/deepseek.gguf

ENV MODEL_PATH=/models/deepseek.gguf

COPY . /app

CMD ["python3", "-u", "handler.py"]
