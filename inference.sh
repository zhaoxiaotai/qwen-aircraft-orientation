#!/bin/bash
    CUDA_VISIBLE_DEVICES=0 \
    IMAGE_MAX_TOKEN_NUM=1024 \
    VIDEO_MAX_TOKEN_NUM=128 \
    FPS_MAX_FRAMES=16 \
    swift infer \
        --adapters /root/autodl-tmp/loras/Qwen/Qwen3-VL-8B-Instruct/v5-20260410-110528/checkpoint-99 \
        --stream true