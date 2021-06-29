#!/bin/bash -l
devices=$1

num=$(echo $1 | awk -F',' '{print NF}')
python=$2
script=$3
args="${@:4}"

echo "HOST    : "$(hostname)
echo "DEVICES : "$devices
echo "NPROC   : "$num
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$devices \
	mpirun -np $num -H localhost:$num -bind-to none -map-by slot \
	-mca pml ob1 -mca btl ^openib \
	-mca orte_base_help_aggregate 0 \
	-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
	$python $script $args
