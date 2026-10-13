#!/bin/bash

# ============================================================
# SAM3 affordance evaluation batch runner
#
# データセット構造:
#
# DATA/
# ├── MeasureData/
# │   └── Sensor/
# │       └── Object/
# │           └── Number/
# │               └── image files
# └── Annotation/
#     └── Sensor/
#         └── Object/
#             └── Number/
#                 └── GT images
#
# prompts_affordance.tsv:
#   object    prompt_id    affordance
#
# affordance_colors.tsv:
#   affordance    r    g    b
#
# このスクリプトは
#   Sensor × Object × Number × Affordance Prompt
# の全条件を自動実行する。
# ============================================================


# -----------------------------
# 基本設定
# -----------------------------

DATA_ROOT="./DATA"
MEASURE_ROOT="$DATA_ROOT/MeasureData"
ANNOTATION_ROOT="$DATA_ROOT/Annotation"

RESULT_ROOT="./results_affordance_SAM3"

PROMPT_FILE="./prompts_affordance.tsv"
AFFORDANCE_COLOR_FILE="./affordance_colors.tsv"

PYTHON_SCRIPT="./test_IoU_affordance_SAM3.py"

THRESHOLD=0.5


# -----------------------------
# 必要ファイルの存在確認
# -----------------------------

if [ ! -d "$MEASURE_ROOT" ]; then
    echo "[error] MeasureData directory not found: $MEASURE_ROOT"
    exit 1
fi

if [ ! -d "$ANNOTATION_ROOT" ]; then
    echo "[error] Annotation directory not found: $ANNOTATION_ROOT"
    exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
    echo "[error] Prompt TSV not found: $PROMPT_FILE"
    exit 1
fi

if [ ! -f "$AFFORDANCE_COLOR_FILE" ]; then
    echo "[error] Affordance color TSV not found: $AFFORDANCE_COLOR_FILE"
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

for sensor_dir in "$MEASURE_ROOT"/*; do

    [ -d "$sensor_dir" ] || continue

    sensor=$(basename "$sensor_dir")


    for object_dir in "$sensor_dir"/*; do

        [ -d "$object_dir" ] || continue

        object_name=$(basename "$object_dir")


        for count_dir in "$object_dir"/*; do

            [ -d "$count_dir" ] || continue

            num_objects=$(basename "$count_dir")

            # MeasureData と同じ階層関係から Annotation 側GTパスを作る
            gt_dir="$ANNOTATION_ROOT/$sensor/$object_name/$num_objects"

            if [ ! -d "$gt_dir" ]; then
                echo "[skip] GT directory not found: $gt_dir"
                continue
            fi


            prompt_found=false

            # ------------------------------------------------
            # prompts_affordance.tsv
            #
            # object    prompt_id    affordance
            #
            # 現在の object に対応する全affordance promptを実行する
            # ------------------------------------------------
            while IFS=$'\t' read -r object prompt_id affordance; do

                # WindowsのCRLF対策
                affordance="${affordance%$'\r'}"

                # ヘッダ・空行を無視
                [ "$object" = "object" ] && continue
                [ -z "$object" ] && continue

                # 現在のobjectだけ処理
                [ "$object" = "$object_name" ] || continue

                prompt_found=true

                if [ -z "$prompt_id" ] || [ -z "$affordance" ]; then
                    echo "[skip] Invalid prompt entry for object: $object_name"
                    continue
                fi

                result_dir="$RESULT_ROOT/$sensor/$object_name/$num_objects/$prompt_id"
                mkdir -p "$result_dir"

                echo
                echo "============================================================"
                echo "Method            : SAM3-affordance"
                echo "Sensor            : $sensor"
                echo "Object            : $object_name"
                echo "Number of objects : $num_objects"
                echo "Prompt ID         : $prompt_id"
                echo "Affordance        : $affordance"
                echo "Image directory   : $count_dir"
                echo "GT directory      : $gt_dir"
                echo "Result directory  : $result_dir"
                echo "============================================================"

                python "$PYTHON_SCRIPT" \
                    --image_dir "$count_dir" \
                    --gt_dir "$gt_dir" \
                    --affordance "$affordance" \
                    --affordance_colors "$AFFORDANCE_COLOR_FILE" \
                    --prompt_id "$prompt_id" \
                    --camera_type "$sensor" \
                    --object_name "$object_name" \
                    --num_objects "$num_objects" \
                    --threshold "$THRESHOLD" \
                    --output_dir "$result_dir/overlay" \
                    --csv_out "$result_dir/iou_results.csv"

                status=$?

                if [ "$status" -ne 0 ]; then
                    echo "[error] Evaluation failed:"
                    echo "        Sensor=$sensor"
                    echo "        Object=$object_name"
                    echo "        Count=$num_objects"
                    echo "        Prompt=$prompt_id"
                    echo "        Affordance=$affordance"
                fi

            done < "$PROMPT_FILE"


            if [ "$prompt_found" = false ]; then
                echo "[skip] No affordance prompt found for object: $object_name"
            fi

        done
    done
done

echo
echo "All SAM3 affordance evaluations finished."
