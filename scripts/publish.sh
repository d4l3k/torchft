#!/bin/bash

set -ex

docker build -t torchft .
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.12
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.11
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.10
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.9
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.8
python3 -m twine upload target/wheels/*manylinux2014*
