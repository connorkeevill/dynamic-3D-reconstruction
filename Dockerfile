FROM lileee/ubuntu-16.04-cuda-9.0-python-3.5-pytorch:latest
WORKDIR /app

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt clean
RUN apt update
RUN apt install -y git
RUN apt install -y libeigen3-dev
RUN apt install -y libopencv-dev
RUN apt install -y python3-pip
RUN apt install -y nvidia-cuda-gdb
RUN apt install -y gdb

WORKDIR /app
COPY . .

ARG CKDOCKER=1

RUN cmake .
RUN make -j$(nproc)

CMD ["python", "run.py"]
