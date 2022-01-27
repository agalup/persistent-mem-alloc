#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/home/agnes/mps/mps
export CUDA_MPS_LOG_DIRECTORY=/home/agnes/mps/log

nvidia-cuda-mps-control -d

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate myenv

python pmm.py 32 1000 1
