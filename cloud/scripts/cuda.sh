#!/bin/bash
set -eux

apt-get -y install libcupti-dev linux-image-extra-virtual
cd /tmp/

curl -LO https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
sh cuda_8.0.44_linux-run --silent --driver --toolkit

curl -LO https://nelhage-ml.s3.amazonaws.com/sw/cudnn-8.0-linux-x64-v5.1.tgz
tar -C /usr/local -xzf "cudnn-8.0-linux-x64-v5.1.tgz"
