#!/bin/bash
set -e


# Chinese
models=("cogagent24" "ariaui" "uground" "osatlas-7b" "osatlas-4b" "showui" "seeclick" "qwen1vl" "qwen2vl" "minicpmv" "cogagent" "gpt4o" )

for model in "${models[@]}"
do
    python eval_screenspot_pro.py  \
        --model_type ${model}  \
        --screenspot_imgs "../data/ScreenSpot-Pro/images"  \
        --screenspot_test "../data/ScreenSpot-Pro/annotations"  \
        --task "all" \
        --language "cn" \
        --gt_type "positive" \
        --log_path "./results/cn/${model}.json" \
        --inst_style "instruction"

done