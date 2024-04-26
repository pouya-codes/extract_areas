# Extract Areas from WSI Slides

This script extracts areas from WSI slides based on the provided HistoQC masks.

## Usage

```bash
python __init__.py --slides_path /path/to/slides --masks_path /path/to/masks --slide_down_sample_rate 5 --overlay_down_sample_rate 5 --output_path /path/to/output deepliif --model_dir /path/to/model --tile_size 512
```


## Arguments

- `--slides_path`: Path to the WSI slides location.
- `--masks_path`: Path to the HistoQC masks location.
- `--slide_down_sample_rate`: The rate of down sampling the extracted regions. Default is 5.
- `--overlay_down_sample_rate`: The rate of down sampling to generated overlay. Default is 5.
- `--output_path`: Path to the output location.

### Deepliif Arguments

- `--model_dir`: Path to the unserilized model directory.
- `--tile_size`: Size of the tiles to be processed. Default is 512.


