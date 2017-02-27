#!/bin/bash
set -eux

apt-get update
apt-get -y upgrade
sudo apt-get -y install build-essential virtualenv cmake zlib1g-dev linux-tools-common linux-tools-generic
shutdown -r now
