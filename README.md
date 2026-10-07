# Study Foundation Model

This repository contains evaluation utilities for segmentation foundation models.

The current workflow evaluates **text-prompted SAM3 segmentation masks** using **Intersection over Union (IoU)** and supports batch execution over multiple experimental conditions.

## Files

```text
.
├── test_IoU.py
├── run_all.sh
├── prompts.tsv
└── README.md
```

- `test_IoU.py`: Runs SAM3 inference and IoU evaluation.
- `run_all.sh`: Automatically runs all existing dataset conditions.
- `prompts.tsv`: Defines one or more text prompts for each object.

## Dataset Structure

`Data` and `GT` are assumed to have the same hierarchy.

```text
Data/
├── Sensor1/
│   ├── objectA/
│   │   ├── 1/
│   │   └── 4/
│   └── objectB/
│       └── 1/
└── Sensor2/
    └── ...
```

```text
GT/
├── Sensor1/
│   ├── objectA/
│   │   ├── 1/
│   │   └── 4/
│   └── objectB/
│       └── 1/
└── Sensor2/
    └── ...
```

Each level corresponds to:

```text
Sensor / Object / Number of objects
```

For example:

```text
Data/RealSense/cup/4/
```

is paired automatically with:

```text
GT/RealSense/cup/4/
```

## Multiple Prompts for One Object

Multiple prompts can be tested for the same object by using `prompts.tsv`.

Example:

```text
object	prompt_id	prompt
cup	p01	mug
cup	p02	cup
cup	p03	drinking cup
screw	p01	screw
screw	p02	metal screw
```

The file is tab-separated.

| Column | Description |
|---|---|
| `object` | Object directory name |
| `prompt_id` | Short identifier used for organizing results |
| `prompt` | Text prompt passed to SAM3 |

For example, the object `cup` is evaluated three times with:

```text
mug
cup
drinking cup
```

Using `prompt_id` prevents different prompt results from overwriting each other.

## Batch Evaluation

`run_all.sh` evaluates all existing combinations of:

```text
Sensor × Object × Number of objects × Prompt
```

The script reads the directory structure under `Data/`, finds the corresponding GT path, and then runs every prompt registered for that object.

Nonexistent dataset combinations are not generated.

If a corresponding GT directory is missing, that condition is skipped.

If an object has no entry in `prompts.tsv`, that object is skipped.

## Configuration

The following values can be changed at the top of `run_all.sh`:

```bash
DATA_ROOT="./Data"
GT_ROOT="./GT"
RESULT_ROOT="./results"
PROMPT_FILE="./prompts.tsv"
SEGMENTATION_TYPE="semantic"
PYTHON_SCRIPT="./test_IoU.py"
```

`SEGMENTATION_TYPE` should be either:

```text
semantic
```

or:

```text
instance
```

## Running the Shell Script

Give the script execution permission once:

```bash
chmod +x run_all.sh
```

Then run:

```bash
./run_all.sh
```

Alternatively:

```bash
bash run_all.sh
```

## Example Command

For the condition:

```text
Sensor      = RealSense
Object      = cup
Num objects = 4
Prompt      = drinking cup
```

the script executes a command equivalent to:

```bash
python test_IoU.py \
    --image_dir "./Data/RealSense/cup/4" \
    --gt_dir "./GT/RealSense/cup/4" \
    --segmentation_type semantic \
    --prompt "drinking cup" \
    --camera_type "RealSense" \
    --object_name "cup" \
    --num_objects "4" \
    --output_dir "./results/RealSense/cup/4/p03/overlay" \
    --csv_out "./results/RealSense/cup/4/p03/iou_results.csv"
```

If your local `test_IoU.py` uses different command-line option names, edit the corresponding options in `run_all.sh`.

## Output Structure

Results are saved separately for each prompt.

```text
results/
└── RealSense/
    └── cup/
        └── 4/
            ├── p01/
            │   ├── overlay/
            │   └── iou_results.csv
            ├── p02/
            │   ├── overlay/
            │   └── iou_results.csv
            └── p03/
                ├── overlay/
                └── iou_results.csv
```

This structure makes it easy to compare different prompts under the same experimental condition.

## Evaluation Notes

The IoU evaluation uses the segmentation masks predicted from the text prompt.

Bounding boxes and SAM3 confidence scores are not required for the IoU evaluation itself.

The public repository can document the evaluation interface and workflow without including confidential dataset-specific implementation details.
