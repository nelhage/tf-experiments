#!/bin/bash
set -eux

apt-get update
sudo apt-get -y install build-essential virtualenv cmake zlib1g-dev
