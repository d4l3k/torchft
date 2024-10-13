#!/bin/bash

set -ex

docker build -t torchft .
docker run --rm -v $(pwd):/io torchft build --release
python3 -m twine upload target/wheels/*manylinux2014*