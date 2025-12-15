#!/bin/bash

# 多GPU训练命令示例
# 使用所有可用GPU
#accelerate launch --multi_gpu train_implicit_mlp_attn.py

# 或者指定使用的GPU数量
 accelerate launch --num_processes=4 train_implicit_mlp_attn.py > seedgpu.log
