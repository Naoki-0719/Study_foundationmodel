# Study Foundation Model

This repository contains evaluation code for studying segmentation foundation models.

## SAM3 IoU Evaluation

`test_IoU.py` evaluates segmentation masks predicted by SAM3 from a text prompt using **Intersection over Union (IoU)**.

The script supports two evaluation modes:

- **Semantic segmentation**
- **Instance segmentation**

Bounding boxes and SAM3 confidence scores are not used for the evaluation.

## Evaluation Flow

1. Load an input image.
2. Run SAM3 with a text prompt.
3. Obtain the predicted masks.
4. Binarize the predicted masks.
5. Load the corresponding ground-truth mask(s).
6. Calculate IoU.

## Semantic Segmentation

For semantic segmentation, all masks predicted from the same text prompt are merged into a single binary mask.

For example, if the prompt is:

```text
screw
```

and SAM3 predicts three screw masks, the masks are combined using a logical OR operation.

The merged prediction is then compared with the semantic ground-truth mask.

### Ground-truth structure

```text
gt/
└── semantic/
    ├── image001.png
    ├── image002.png
    └── ...
```

## Instance Segmentation

For instance segmentation, each predicted mask is compared with each ground-truth instance mask.

The script creates an IoU matrix and matches predicted and ground-truth instances one-to-one based on IoU.

Unmatched predictions or ground-truth instances are treated as IoU = 0 when calculating the image-level mean IoU.

### Ground-truth structure

```text
gt/
└── instance/
    ├── image001/
    │   ├── 000.png
    │   ├── 001.png
    │   └── ...
    ├── image002/
    │   ├── 000.png
    │   └── ...
    └── ...
```

## Usage

### Semantic segmentation

```bash
python test_IoU.py \
    --image_dir ./images \
    --gt_dir ./gt \
    --segmentation_type semantic \
    --prompt "screw"
```

### Instance segmentation

```bash
python test_IoU.py \
    --image_dir ./images \
    --gt_dir ./gt \
    --segmentation_type instance \
    --prompt "screw"
```

## Main Arguments

| Argument | Description |
|---|---|
| `--image_dir` | Directory containing input images |
| `--gt_dir` | Root directory containing ground-truth masks |
| `--segmentation_type` | `semantic` or `instance` |
| `--prompt` | Text prompt given to SAM3 |
| `--output_dir` | Directory for overlay images |
| `--csv_out` | Output CSV file for IoU results |
| `--threshold` | Threshold used to binarize predicted masks |

## Output

The script outputs:

- IoU for each evaluated image
- Mean IoU over the dataset
- Number of predicted and ground-truth instances
- Overlay images of the predicted masks
- A CSV file containing the evaluation results

## Notes

This repository is intended for experimental evaluation of segmentation foundation models.

The current implementation focuses on evaluating **text-prompted SAM3 segmentation masks** using IoU.
