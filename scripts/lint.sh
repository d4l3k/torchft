#!/bin/bash

set -ex

cargo fmt
black .
pyre
