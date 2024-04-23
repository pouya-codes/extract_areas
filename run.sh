#!/bin/bash
#SBATCH --job-name DeepLIIF
#SBATCH --cpus-per-task 20
#SBATCH --chdir /projects/ovcare/classification/pouya/QA-QC/DeepLIIF
#SBATCH --output /projects/ovcare/classification/pouya/QA-QC/scripts/DeepLIIF/logs/test.out
#SBATCH --error /projects/ovcare/classification/pouya/QA-QC/scripts/DeepLIIF/logs/test.err
#SBATCH -p gpu2080
#SBATCH --time=1:00:00
#SBATCH --mem=50G
source /projects/ovcare/classification/pouya/miniconda3/etc/profile.d/conda.sh
conda activate deepliif_env

SLIDE_PATH=/projects/ovcare/classification/pouya/QA-QC/datasets/Images/
MASKS_PATH=
OUTPUT_PATH=
python --slides_path $SLIDE_PATH --masks_path $MASKS_PATH --output_path $OUTPUT_PATH