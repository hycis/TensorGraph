#!/bin/bash -l
module purge
module load shared
module load python/3.6.6
module load ml-python3deps/1.0.0
module load tensorflow/1.8.0-ref
module load horovod/0.14.1

python=python3

devices=$1

num=$(echo $1 | awk -F',' '{print NF}')
script=$2
args="${@:3}"

echo "HOST    : "$(hostname)
echo "DEVICES : "$devices
echo "NPROC   : "$num
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$devices $python $script $args
