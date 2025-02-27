#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd ../

python pipeline.py \
    -r "$runname-idev-32shot" \
    -d seed \
    -m idefics-9b \
    -e \
    -s 0 \
    --requires_memory 40000 \
    --eval-args "eval.ckpt_epochs=null eval.batch_size=2 eval.num_shot=32"