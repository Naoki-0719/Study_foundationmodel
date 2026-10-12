#!/bin/bash

# ============================================================
# PaDT IoU evaluation batch runner
#
# Data と GT が同じ階層構造であることを前提に、
# Sensor × Object × Number of objects × Prompt
# を自動実行する。
#
# prompts_PaDT.tsv には長いPaDT命令文を直接書かず、
# object / prompt_id / prompt_type / target
# のみを記述する。
# ============================================================


# -----------------------------
# 基本設定
# -----------------------------

DATA_ROOT="./Data"
GT_ROOT="./GT"
RESULT_ROOT="./results_PaDT"
PROMPT_FILE="./prompts_PaDT.tsv"

# semantic または instance
SEGMENTATION_TYPE="instance"

PYTHON_SCRIPT="./test_IoU_PaDT.py"
MODEL_PATH="PaDT-MLLM/PaDT_Pro_3B"
DEVICE_INDEX=0
MAX_SIDE=644
THRESHOLD=0.5


# -----------------------------
# 必要ファイルの存在確認
# -----------------------------

if [ ! -d "$DATA_ROOT" ]; then
    echo "[error] Data directory not found: $DATA_ROOT"
    exit 1
fi

if [ ! -d "$GT_ROOT" ]; then
    echo "[error] GT directory not found: $GT_ROOT"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "[error] Prompt file not found: $PROMPT_FILE"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[error] Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

mkdir -p "$RESULT_ROOT"


# ============================================================
# Sensor -> Object -> Number -> Prompt
# ============================================================

for sensor_dir in "$DATA_ROOT"/*; do

    [ -d "$sensor_dir" ] || continue
    sensor=$(basename "$sensor_dir")

    for object_dir in "$sensor_dir"/*; do

        [ -d "$object_dir" ] || continue
        object_name=$(basename "$object_dir")

        for count_dir in "$object_dir"/*; do

            [ -d "$count_dir" ] || continue
            num_objects=$(basename "$count_dir")

            # Dataと同じ階層関係からGT pathを作成
            gt_dir="$GT_ROOT/$sensor/$object_name/$num_objects"

            if [ ! -d "$gt_dir" ]; then
                echo "[skip] GT directory not found: $gt_dir"
                continue
            fi

            prompt_found=false

            # ------------------------------------------------
            # prompts_PaDT.tsv
            # object    prompt_id    prompt_type    target
            #
            # 現在のobjectに対応する全Promptを実行する。
            # ------------------------------------------------
            while IFS=$'\t' read -r object prompt_id prompt_type target; do

                # Windows CRLF対策
                target="${target%$'\r'}"

                # header / empty lineを無視
                [ "$object" = "object" ] && continue
                [ -z "$object" ] && continue

                # 現在のobject以外は無視
                [ "$object" = "$object_name" ] || continue

                prompt_found=true

                if [ -z "$prompt_id" ] || [ -z "$prompt_type" ] || [ -z "$target" ]; then
                    echo "[skip] Invalid prompt entry for object: $object_name"
                    continue
                fi

                # Promptごとに結果を分けて保存
                result_dir="$RESULT_ROOT/$sensor/$object_name/$num_objects/$prompt_id"
                mkdir -p "$result_dir"

                echo
                echo "============================================================"
                echo "Method            : PaDT"
                echo "Sensor            : $sensor"
                echo "Object            : $object_name"
                echo "Number of objects : $num_objects"
                echo "Prompt ID         : $prompt_id"
                echo "Prompt type       : $prompt_type"
                echo "Target            : $target"
                echo "Image directory   : $count_dir"
                echo "GT directory      : $gt_dir"
                echo "Result directory  : $result_dir"
                echo "============================================================"

                python "$PYTHON_SCRIPT" \
                    --image_dir "$count_dir" \
                    --gt_dir "$gt_dir" \
                    --segmentation_type "$SEGMENTATION_TYPE" \
                    --prompt_type "$prompt_type" \
                    --target "$target" \
                    --model_path "$MODEL_PATH" \
                    --device_index "$DEVICE_INDEX" \
                    --max_side "$MAX_SIDE" \
                    --threshold "$THRESHOLD" \
                    --camera_type "$sensor" \
                    --object_name "$object_name" \
                    --num_objects "$num_objects" \
                    --output_dir "$result_dir/overlay" \
                    --csv_out "$result_dir/iou_results.csv" \
                    --instance_csv_out "$result_dir/instance_iou_results.csv"

                status=$?

                if [ "$status" -ne 0 ]; then
                    echo "[error] PaDT evaluation failed:"
                    echo "        Sensor=$sensor"
                    echo "        Object=$object_name"
                    echo "        Count=$num_objects"
                    echo "        Prompt=$prompt_id"
                fi

            done < "$PROMPT_FILE"

            if [ "$prompt_found" = false ]; then
                echo "[skip] No PaDT prompt found for object: $object_name"
            fi

        done
    done
done


echo
echo "All PaDT evaluations finished."
