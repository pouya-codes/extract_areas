# WSI Slide Area Extractor

This project extracts areas/cores from Whole Slide Images (WSI) and process them based on Deepliif moldel.

## Requirements

- Conda
- Deepliif
- Segment Anything

## Install

To create the running enviroment use the following command:

```shell
conda create --name deepliif_env --file requirements.txt
```

Download Deepliif latest model by following this link:
```
https://zenodo.org/record/4751737#.YKRTS0NKhH4
```

For generation the masks download lastest segment anything model checkpoint from the following link:

```
https://github.com/facebookresearch/segment-anything#model-checkpoints
```

## Usage

To use this script, you need to pass several arguments:

- `--slides_path`: Path to the WSI slides location (required)
- `--masks_path`: Path to the HistoQC masks location (optional)
- `--annotations_path`: Path to the slides' annotation (optional)
- `--output_path`: Path to the output location (required)
- `--slide_down_sample_rate`: The rate of down sampling the extracted regions (default: 5)
- `--overlay_down_sample_rate`: The rate of down sampling to generated overlay (default: 5)

You can also choose to run two additional functionalities:

- `--deepliif`: Run deepliif (optional)
  - `--model_dir`: Path to the unserilized model directory (required if `--deepliif` is set)
  - `--tile_size`: Size of the tiles to be processed (default: 256)
  - `--post_processing`: Run postprocessing algorithm on the results (default: True)

- `--mask_generator`: Run mask generator (optional)
  - `--model_path`: Path to the segment anything model checkpoint (required if `--mask_generator` is set)

Example command:

```shell
conda activate deepliif_env
python myparser.py --slides_path "path/to/slides" --output_path "path/to/output" --deepliif --model_dir "path/to/model" --mask_generator --model_path "path/to/checkpoint"
```