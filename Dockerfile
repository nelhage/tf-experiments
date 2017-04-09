FROM ubuntu:16.04

RUN apt-get update && apt-get -y install python3 python3-pip virtualenv
RUN virtualenv --python=python3 /venv

RUN apt-get update && apt-get -y install cmake build-essential
RUN apt-get update && apt-get -y install zlib1g-dev

RUN mkdir /src
ADD requirements.txt /src/requirements.txt

WORKDIR /src

RUN /venv/bin/pip install -r requirements.txt

ADD . /src
