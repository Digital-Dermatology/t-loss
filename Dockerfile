FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

MAINTAINER Alvaro Gonzalez Jimenez <alvaro.gonzalezjimenez@unibas.ch>

RUN apt-get update && apt install -y vim git
RUN python -m pip install --upgrade pip

COPY . /app/

WORKDIR /app/

RUN pip install -U -r requirements.txt

VOLUME /data
CMD ["/bin/bash"]
