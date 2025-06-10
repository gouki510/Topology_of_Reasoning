#!/bin/sh

## GSM8K
export HF_TOKEN=your_token
CSV_PATH=your_path
DATASET=gsm8k #gsm8k, math_500, aime
MODEL=your_model  #deepseek-ai/DeepSeek-R1-Distill-Qwen-32B, Qwen/Qwen2.5-32B
NUM_TYPES=200
TARGET_LAYER_RATIO=0.9 #0.1, 0.3, 0.5, 0.7, 0.9

python src/cluster_steps_generated.py \
    --model_name_or_path $MODEL \
    --tokenizer_name_or_path $MODEL \
    --batch_size 8 \
    --dataset $DATASET \
    --num_types $NUM_TYPES \
    --df_path $CSV_PATH \
    --target_layer_ratio $TARGET_LAYER_RATIO