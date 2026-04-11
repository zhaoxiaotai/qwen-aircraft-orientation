#!/bin/bash
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    IMAGE_MAX_TOKEN_NUM=1024 \
    VIDEO_MAX_TOKEN_NUM=128 \
    FPS_MAX_FRAMES=16 \
    NPROC_PER_NODE=1 \
    CUDA_VISIBLE_DEVICES=0 \
    swift sft \
    --model '/root/autodl-tmp/models/Qwen/Qwen3-VL-8B-Instruct' \
    --dataset /root/autodl-tmp/mq_9b/train.jsonl \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --attn_impl sdpa \
    --padding_free false \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --packing false \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing false \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /root/autodl-tmp/loras/Qwen/Qwen3-VL-8B-Instruct \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4