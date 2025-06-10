#!/bin/sh

DATASET=gsm8k #gsm8k, math_500, aime
MODEL=your_model #deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, Qwen/Qwen2.5-32B
EFFICIENT=none #none, parameter_efficient
FLASH=True #True, False
export HF_TOKEN=your_token
TMP_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT_DIR=eval_result/gsm8k/$MODEL/${TMP_TIME}
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --base_model_name_or_path $MODEL \
    --model_name_or_path $MODEL \
    --parameter_efficient_mode $EFFICIENT \
    --dataset $DATASET \
    --batch_size 1 \
    --max_length 8192 \
    --seed 100 \
    --load_in_8bit True \
    --flash_attention $FLASH \
    --num_test 1000 \
    --output_dir $OUTPUT_DIR \
