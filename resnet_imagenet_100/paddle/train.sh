#!/bin/bash
# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -e

# data add:
# hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310/app/idl/users/dl/dangqingqing/data/dataset_100

config=resnet_torch.py

paddle train \
  --config=$config \
  --log_period=20 \
  --test_all_data_in_one_period=1 \
  --use_gpu=1 \
  --test_period=400 \
  --trainer_count=4 \
  --num_passes=106 \
  --save_dir=./output \
  --cudnn_dir=/home/dangqingqing/tools/cudnn-5.1/lib64 \
  >train.log 2>&1 &
