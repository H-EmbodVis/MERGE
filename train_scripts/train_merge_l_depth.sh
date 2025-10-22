#!/bin/bash

accelerate launch --num_processes=8 --main_process_port=36661 ./unideg/train_merge_large_depth.py \
 --mixed_precision=bf16 --train_batch_size=1 --dataloader_num_workers=8 --pretrained_model_name_or_path PATH/FLUX.1-dev \
--gradient_accumulation_steps=4 --use_8bit_adam --learning_rate=3e-4 --lr_scheduler="linear" --checkpoints_total_limit 1 \
--max_train_steps=30000 --validation_steps 5000 --checkpointing_steps 5000  --output_dir=./outputs/merge_large_depth_b32_30k




