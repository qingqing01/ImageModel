#!/bin/bash
set -e

export PATH=/home/dangqingqing/caffe/build/tools/:$PATH

if [ ! -d "logs" ]; then
  mkdir logs
fi

caffe train --solver=solver.prototxt -gpu 0,1,2,3 > logs/googlenet-4gpu.log 2>&1 &
#caffe test --model=train_val.prototxt \
#           --weights=./model/imagenet-100_iter_75000.caffemodel \
#           -gpu all --iterations=50 > logs/googlenet-test.log 2>&1 &
