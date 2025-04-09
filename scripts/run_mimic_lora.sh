#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')

cd ../src/

python pipeline.py \
    -r "$runname-r" \
    -d boolq \
    -m llama-7b \
    -q 1000 \
    -s 32 \
    --train-args "encoder=mimic peft=mini_lora " \
    --eval-args "encoder=mimic peft=mini_lora "
