#!/bin/bash

# ============================================================
# SAM3 IoU evaluation batch runner
#
# Data と GT が同じ階層構造になっていることを前提に、
# Sensor × Object × Number of objects × Prompt
# の全条件を自動実行する。
# ============================================================

DATA_ROOT="./Data"
GT_ROOT="./GT"
RESULT_ROOT="./results"
PROMPT_FILE="./prompts.tsv"
SEGMENTATION_TYPE="semantic"
PYTHON_SCRIPT="./test_IoU.py"

# 必要なファイル・ディレクトリの確認
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

# Sensor -> Object -> Number of objects の順に走査
for sensor_dir in "$DATA_ROOT"/*; do
    [ -d "$sensor_dir" ] || continue
    sensor=$(basename "$sensor_dir")

    for object_dir in "$sensor_dir"/*; do
        [ -d "$object_dir" ] || continue
        object_name=$(basename "$object_dir")

        for count_dir in "$object_dir"/*; do
            [ -d "$count_dir" ] || continue
            num_objects=$(basename "$count_dir")

            # Data と同じ階層関係から GT パスを作る
            gt_dir="$GT_ROOT/$sensor/$object_name/$num_objects"

            if [ ! -d "$gt_dir" ]; then
                echo "[skip] GT directory not found: $gt_dir"
                continue
            fi

            prompt_found=false

            # 現在の Object に対応する Prompt をすべて実行
            while IFS=$'\t' read -r object prompt_id prompt; do
                # Windows の CRLF 対策
                prompt="${prompt%$'\r'}"

                # ヘッダと空行を無視
                [ "$object" = "object" ] && continue
                [ -z "$object" ] && continue

                # 現在処理中の Object と一致する行だけを使う
                [ "$object" = "$object_name" ] || continue

                prompt_found=true

                if [ -z "$prompt_id" ] || [ -z "$prompt" ]; then
                    echo "[skip] Invalid prompt entry for object: $object_name"
                    continue
                fi

                # Prompt ごとに結果ディレクトリを分ける
                result_dir="$RESULT_ROOT/$sensor/$object_name/$num_objects/$prompt_id"
                mkdir -p "$result_dir"

                echo
                echo "============================================================"
                echo "Sensor            : $sensor"
                echo "Object            : $object_name"
                echo "Number of objects : $num_objects"
                echo "Prompt ID         : $prompt_id"
                echo "Prompt            : $prompt"
                echo "Image directory   : $count_dir"
                echo "GT directory      : $gt_dir"
                echo "Result directory  : $result_dir"
                echo "============================================================"

                python "$PYTHON_SCRIPT" \
                    --image_dir "$count_dir" \
                    --gt_dir "$gt_dir" \
                    --segmentation_type "$SEGMENTATION_TYPE" \
                    --prompt "$prompt" \
                    --camera_type "$sensor" \
                    --object_name "$object_name" \
                    --num_objects "$num_objects" \
                    --output_dir "$result_dir/overlay" \
                    --csv_out "$result_dir/iou_results.csv"

                status=$?

                if [ "$status" -ne 0 ]; then
                    echo "[error] Evaluation failed:"
                    echo "        Sensor=$sensor Object=$object_name Count=$num_objects Prompt=$prompt_id"
                fi

            done < "$PROMPT_FILE"

            if [ "$prompt_found" = false ]; then
                echo "[skip] No prompt found for object: $object_name"
            fi
        done
    done
done

echo
echo "All evaluations finished."
