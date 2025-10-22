#!/bin/bash

accelerate launch --num_processes=8 --main_process_port=36661 merge/train_merge_base_depth.py \
--dataloader_num_workers=8 --max_train_steps 30000 --learning_rate 1e-4 --train_batch_size 4 \
--validation_steps 1000 --checkpointing_steps 5000 --checkpoints_total_limit 1 \
--pretrained_model_name_or_path PATH/PixArt-XL-2-512x512 \
--output_dir=./outputs/merge_base_depth_b32_30k





