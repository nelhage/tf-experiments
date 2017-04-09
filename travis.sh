#!/bin/bash
set -eux

version=$(git rev-parse HEAD | head -c10)
docker build . -t "nelhage/ml:$version"
docker push "nelhage/ml:$version"
