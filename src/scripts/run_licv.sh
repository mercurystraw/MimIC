#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd ../

python pipeline.py \
    -r "$runname" \
    -d ocr_vqa \
    -m idefics2-8b-base \
    -q 1000 \
    -s 8 \
    --train-args "encoder=licv peft=mimic +peft.scale_lr=1e-2 training.epochs=10" \
    --eval-args "encoder=licv peft=mimic eval.ckpt_epochs=9 eval.batch_size=4"