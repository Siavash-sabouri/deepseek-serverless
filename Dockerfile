FROM runpod/llama-cpp-python:latest

COPY . /app
WORKDIR /app

RUN pip install --upgrade llama-cpp-python

CMD ["handler.handler"]
