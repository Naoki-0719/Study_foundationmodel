#!/bin/bash

# ============================================================
# PaDT single-load batch evaluation
#
# PaDTモデルを1回だけロードして、
# Sensor × Object × Number × Prompt
# の全条件をPython内で連続実行する。
# ============================================================

python run_all_PaDT_singleload.py \
    --data_root "./Data" \
    --gt_root "./GT" \
    --result_root "./results_PaDT_singleload" \
    --prompt_file "./prompts_PaDT.tsv" \
    --segmentation_type "instance" \
    --model_path "PaDT-MLLM/PaDT_Pro_3B" \
    --device_index 0 \
    --max_side 644 \
    --threshold 0.5
