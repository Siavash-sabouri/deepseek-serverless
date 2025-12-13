FROM python:3.10

WORKDIR /app

RUN apt update && apt install -y \
    git \
    curl \
    build-essential \
    cmake \
    && pip install --no-cache-dir runpod llama-cpp-python

COPY . /app

CMD ["python3", "-u", "handler.py"]
