#!/bin/bash

runname=$(basename "$0" .sh | sed 's/^run_//' | sed 's/_/-/g')

cd ../src/

# -s should be 0 for ICL.
# to specify the number of shots, use --eval-args "eval.num_shot=32"
python pipeline.py \
    -r "$runname-llava-0shot" \
    -d vqav2,ok_vqa,coco \
    -m llava-interleave-7b \
    -e \
    -s 0 \
    --requires_memory 40000 \
    --eval-args "eval.ckpt_epochs=null eval.batch_size=4 eval.num_shot=0 eval.query_set_size=1000"