#!/usr/bin/env bash
# Example: ./mujoco_evaluate.sh <env_id> <model_path> <num_trajs> <iter_num> <video_len>

cd ..

python main.py \
    --env_id=$1 \
    --seed=1 \
    --log_dir="data/logs" \
    --task="evaluate" \
    --algo="ddpg" \
    --num_trajs=$3 \
    --iter_num=$4 \
    --no-render \
    --no-record \
    --video_len=$5 \
    --video_dir="data/videos" \
    --model_path=$2 \
    --with_layernorm
