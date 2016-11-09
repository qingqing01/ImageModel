#!/bin/bash
set -e

export PATH=/home/dangqingqing/caffe/build/tools/:$PATH

if [ ! -d "data" ]; then

fi

if [ ! -d "logs" ]; then
  mkdir logs
fi

caffe train --solver=solver.prototxt -gpu all > logs/resnet-4gpu-batch128.log 2>&1 &
#caffe test --model=train_val.prototxt \
#           --weights=./model/imagenet-100_iter_75000.caffemodel \
#           -gpu all --iterations=50 > logs/resnet-test.log 2>&1 &
#caffe train --solver=solver.prototxt --snapshot=./model/imagenet-100_iter_40000.solverstate -gpu all > logs/resnet-4gpu-resume.log 2>&1 &
