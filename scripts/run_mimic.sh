#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')

cd ../src/

python pipeline.py \
    -r "$runname" \
    -d boolq \
    -m llama-7b \
    -q 500 \
    -s 16 \
    --devices 0,1,2,3 \
    --requires_memory 20000 \
    --wait-devices-timeout 100000 \
    --train-args "encoder=mimic peft=mimic" \
    --eval-args "encoder=mimic peft=mimic"
