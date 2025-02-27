#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')

cd ../

python pipeline.py \
    -r "$runname" \
    -d coco,ok_vqa,vqav2 \
    -m idefics-9b \
    -q 1000 \
    -s 32 \
    --requires_memory 20000 \
    --train-args "encoder=attn_shift_ffn_mse peft=mimic" \
    --eval-args "encoder=attn_shift_ffn_mse peft=mimic"
