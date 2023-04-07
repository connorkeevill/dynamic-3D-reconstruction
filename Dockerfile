FROM lileee/ubuntu-16.04-cuda-9.0-python-3.5-pytorch:latest
WORKDIR /app

ARG DEBIAN_FRONTEND="noninteractive"

RUN apt clean
RUN apt update
RUN apt install -y git
RUN apt install -y libeigen3-dev
RUN apt install -y libopencv-dev
#RUN apt install -y catkin
RUN apt install -y python3-pip
#RUN pip3 install catkin-tools
#RUN mkdir catkin_ws
#WORKDIR /app/catkin_ws
#RUN mkdir src
#RUN catkin init
#WORKDIR /app/catkin_ws/src
#RUN git clone https://github.com/ros/catkin.git
#RUN mkdir refusion

WORKDIR /app
COPY . .

RUN cmake .
RUN make -j4

CMD ["python", "run.py"]
