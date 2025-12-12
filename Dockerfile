FROM ghcr.io/ggerganov/llama.cpp:latest

RUN apt update && apt install -y python3-pip

COPY . /app
WORKDIR /app

RUN pip3 install --upgrade llama-cpp-python

CMD ["handler.handler"]
