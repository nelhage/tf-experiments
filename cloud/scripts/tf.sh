#!/bin/bash
cd ~ubuntu
virtualenv --python=python3 venv
venv/bin/pip install --upgrade pip
venv/bin/pip install tensorflow tensorflow-gpu
venv/bin/pip install https://storage.googleapis.com/tensorflow/linux/cpu/protobuf-3.2.0-cp35-none-linux_x86_64.whl
