#!/bin/bash
set -eux

apt-get -y install libcupti-dev "linux-image-extra-$(uname -r)"
cd /tmp/

# Driver

driver=--driver
if [ "$(lspci -mmn -d 10de:118a)" ]; then
    echo "Installing for GRID K520"
    curl -LO http://us.download.nvidia.com/XFree86/Linux-x86_64/367.57/NVIDIA-Linux-x86_64-367.57.run
    sh NVIDIA-Linux-x86_64-367.57.run -a -q --ui=none
    driver=
else
    lspci -nn -d 10de:
    echo "Installing most recent driver."
fi

# CUDA

curl -LO https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
sh cuda_8.0.61_375.26_linux-run $driver --silent --toolkit --toolkitpath=/usr/local/cuda-8.0 --verbose

if ! grep -q 'CUDA Version 8' /usr/local/cuda-8.0/version.txt; then
    echo "CUDA install failed!" >&2
    exit 1
fi

ln -s cuda-8.0 /usr/local/cuda

# cuDNN

curl -LO https://nelhage-ml.s3.amazonaws.com/sw/cudnn-8.0-linux-x64-v5.1.tgz
tar -C /usr/local -xzf "cudnn-8.0-linux-x64-v5.1.tgz"

cat > /etc/profile.d/cuda.sh <<'EOF'
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"
EOF


# Blacklist nouveau
cat > /etc/modprobe.d/blacklist-nouveau.conf <<EOF
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
EOF
update-initramfs -k all -u
