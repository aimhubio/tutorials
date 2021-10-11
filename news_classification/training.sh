#!/bin/bash
CUDA_LAUNCH_BLOCKING=1 python training.py \
        --data_path ag_news \
        --prop_split 0.95 \
        --batch_size 64 \
        --device_no 0 \
        --embed_dim 64 \
        --lr 5 \
        --log_interval 500 \
        --max_epoch 10