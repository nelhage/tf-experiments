FROM ubuntu:16.04

RUN apt-get update && apt-get -y install python python-dev python-pip python3 python3-pip virtualenv
RUN apt-get update && apt-get -y install cmake build-essential zlib1g-dev

RUN pip install crcmod
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r /tmp/requirements.txt

ADD https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-183.0.0-linux-x86_64.tar.gz /gcloud.tgz
RUN tar -C /usr/local -xzf /gcloud.tgz
RUN mkdir -p /usr/local/bin
RUN ln -nsf /usr/local/google-cloud-sdk/bin/* /usr/local/bin/
