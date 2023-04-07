FROM lileee/ubuntu-16.04-cuda-9.0-python-3.5-pytorch:latest
WORKDIR /app

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt clean
RUN apt update
RUN apt install -y git
RUN apt install -y libeigen3-dev
RUN apt install -y libopencv-dev
RUN apt install -y python3-pip

WORKDIR /app
COPY . .

RUN cmake .
RUN make -j4

CMD ["python", "run.py"]
