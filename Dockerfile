FROM python:3.10

WORKDIR /app

RUN apt update && apt install -y \
    git \
    curl \
    build-essential \
    cmake \
    && pip install --no-cache-dir runpod llama-cpp-python

# download Q6_K model
RUN mkdir -p /models && \
    curl -L \
    https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/resolve/main/deepseek-coder-6.7b-instruct.Q6_K.gguf \
    -o /models/deepseek.gguf

ENV MODEL_PATH=/models/deepseek.gguf

COPY . /app

CMD ["python3", "-u", "handler.py"]
