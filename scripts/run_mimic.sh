#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')

cd ../src/

python pipeline.py \
    -r "$runname-idev2-lm" \
    -d vqav2 \
    -m idefics2-8b-base \
    -q 1000 \
    -s 8 \
    --devices 0,1,2,3 \
    --requires_memory 40000 \
    --wait-devices-timeout 100000 \
    --train-args "encoder=mimic peft=mimic encoder.model_strategy=Strategy.LM_LOSS resume_train=False" \
    --eval-args "encoder=mimic peft=mimic eval.iterations=200"
