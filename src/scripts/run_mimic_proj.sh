#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd ../

python pipeline.py \
    -r "mimic-adapter-h-64" \
    -d ok_vqa,coco,vqav2 \
    -m idefics-9b \
    -q 1000 \
    -s 32 \
    --train-args "encoder=mimic_adapter peft=mimic training.lr=5e-4" \
    --eval-args "encoder=mimic_adapter peft=mimic"