#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd ../src/

python pipeline.py \
    -r "$runname" \
    -d ocr_vqa \
    -m idefics2-8b-base \
    -q 1000 \
    -s 8 \
    --train-args "encoder=licv peft=mimic" \
    --eval-args "encoder=licv peft=mimic"