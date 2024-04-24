#!/bin/bash
#SBATCH --job-name ExtractAreas
#SBATCH --cpus-per-task 1
#SBATCH --chdir /projects/ovcare/classification/pouya/QA-QC/Codes/extract_areas
#SBATCH --output /projects/ovcare/classification/pouya/QA-QC/scripts/DeepLIIF/logs/temp.out
#SBATCH --error /projects/ovcare/classification/pouya/QA-QC/scripts/DeepLIIF/logs/temp.err
#SBATCH -p gpu3090,gpu3090short,dgxV100,rtx5000,gpuA6000
#SBATCH --time=1:00:00
#SBATCH --mem=10G
source /projects/ovcare/classification/pouya/miniconda3/etc/profile.d/conda.sh
conda activate deepliif_env
# pip install openslide-python

SLIDE_PATH=/projects/ovcare/classification/pouya/QA-QC/datasets/R204_brafv600e/slides
MASKS_PATH=/projects/ovcare/classification/pouya/QA-QC/datasets/R204_brafv600e/masks
OUTPUT_PATH=/projects/ovcare/classification/pouya/QA-QC/datasets/R204_brafv600e/cores

python __init__.py --slides_path $SLIDE_PATH --masks_path $MASKS_PATH --output_path $OUTPUT_PATH