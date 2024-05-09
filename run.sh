#!/bin/bash
#SBATCH --job-name ExtractAreas
#SBATCH --cpus-per-task 1
#SBATCH --chdir /projects/ovcare/classification/pouya/QA-QC/Codes/extract_areas
#SBATCH --output /projects/ovcare/classification/pouya/QA-QC/scripts/DeepLIIF/logs/temp.out
#SBATCH --error /projects/ovcare/classification/pouya/QA-QC/scripts/DeepLIIF/logs/temp.err
#SBATCH -p gpu3090,gpu3090short,dgxV100,rtx5000,gpuA6000
#SBATCH --time=1:00:00
#SBATCH --mem=10G
# source /projects/ovcare/classification/pouya/miniconda3/etc/profile.d/conda.sh
# conda activate deepliif_env

# DeepLIIF_PATH=/home/pouya/Develop/UBC/QA-QC/Codes/DeepLIIF
# pip install openslide-python
# export PYTHONPATH="${PYTHONPATH}:${DeepLIIF_PATH}"

# SLIDE_PATH=/projects/ovcare/classification/pouya/QA-QC/datasets/R204_brafv600e/slides
# MASKS_PATH=/projects/ovcare/classification/pouya/QA-QC/datasets/R204_brafv600e/masks
# OUTPUT_PATH=/projects/ovcare/classification/pouya/QA-QC/datasets/R204_brafv600e/cores

SLIDE_PATH="/home/pouya/Develop/UBC/QA-QC/Datasets/R204brafv600e/Slides"
MASKS_PATH="/home/pouya/Develop/UBC/QA-QC/Datasets/R204brafv600e/Masks"
OUTPUT_PATH="/home/pouya/Develop/UBC/QA-QC/Datasets/R204brafv600e/Results"
MODEL_DIR="/home/pouya/Develop/UBC/QA-QC/Codes/Models/DeepLIIF_Latest_Model"
"C:/Users/pouya/Develop/QA-QC/Datasets/R204brafv600e/Slides"
"C:/Users/pouya/Develop/QA-QC/Datasets/R204brafv600e/Masks"
"C:/Users/pouya/Develop/QA-QC/Datasets/R204brafv600e/Results_new"
"C:/Users/pouya/Develop/QA-QC/DeepLIIF_Latest_Model"
conda activate deepliif_env

python __init__.py --slides_path "C:/Users/pouya/Develop/QA-QC/Datasets/TNP_Array/Slides" --output_path "C:/Users/pouya/Develop/QA-QC/Datasets/TNP_Array/Results" --annotations_path "C:/Users/pouya/Develop/QA-QC/Datasets/TNP_Array/Annotations" --slide_down_sample_rate 2 --overlay_down_sample_rate 4 deepliif --model_dir "C:/Users/pouya/Develop/QA-QC/DeepLIIF_Latest_Model" --tile_size 512
