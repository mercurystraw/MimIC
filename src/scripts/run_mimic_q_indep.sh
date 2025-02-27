#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')

cd ../

python pipeline.py \
    -r "$runname" \
    -d coco,ok_vqa,vqav2 \
    -m idefics-9b \
    -q 1000 \
    -s 32 \
    --devices 0,1,2,3 \
    --train-args "encoder=mimic encoder.cls.attn_strategy=\"ShiftStrategy.USE_VECTOR_IMPL | ShiftStrategy.MULTI_HEAD\"" \
    --eval-args "encoder=mimic encoder.cls.attn_strategy=\"ShiftStrategy.USE_VECTOR_IMPL | ShiftStrategy.MULTI_HEAD\""
