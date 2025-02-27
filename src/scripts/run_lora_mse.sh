#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//')

cd ../

python pipeline.py \
    -r "lora_mse-r-2" \
    -d vqav2,ok_vqa,coco \
    -m idefics-9b \
    -q 1000 \
    -s 32 \
    --train-args "encoder=lora peft=mini_lora peft.r=2 encoder.model_strategy=\"Strategy.LM_LOSS|Strategy.LAYER_WISE_MSE\" encoder.cls.ffn_strategy=ShiftStrategy.RECORD_HIDDEN_STATES peft.ce_loss_weight=0.5 training.lr=5e-3" \
    --eval-args "encoder=lora peft=mini_lora peft.r=2"
