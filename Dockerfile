FROM tensorflow/tensorflow:latest-gpu
LABEL author=pk13055 version=1.1

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt .
RUN pip3 install -r /app/requirements.txt

COPY . .
ENTRYPOINT ["./entrypoint.sh"]

