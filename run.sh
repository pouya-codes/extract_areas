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
conda activate deepliif_env
python __init__.py --slides_path "D:/Develop/UBC/Datasets/TNP_Array/Slides" --output_path "D:/Develop/UBC/Datasets/TNP_Array/Results_new" --slide_down_sample_rate 4 --overlay_down_sample_rate 4 --deepliif --model_dir "D:/Develop/UBC/DeepLIIF_Latest_Model" --tile_size 256 --mask_generator --model_path "D:/Develop/UBC/extract_areas/models/sam_vit_h.pth"

python __init__.py --slides_path "D:/Develop/UBC/Datasets/Breast_TMA/Slides" --output_path "D:/Develop/UBC/Datasets/Breast_TMA/Results_new" --slide_down_sample_rate 4 --overlay_down_sample_rate 4 --deepliif --model_dir "D:/Develop/UBC/DeepLIIF_Latest_Model" --tile_size 256 --mask_generator --model_path "D:/Develop/UBC/extract_areas/models/sam_vit_h.pth"
python __init__.py --slides_path "D:/Develop/UBC/Datasets/R204brafv600e/Slides" --output_path "D:/Develop/UBC/Datasets/R204brafv600e/Results_new" --slide_down_sample_rate 4 --overlay_down_sample_rate 4 --deepliif --model_dir "D:/Develop/UBC/DeepLIIF_Latest_Model" --tile_size 256 --mask_generator --model_path "D:/Develop/UBC/extract_areas/models/sam_vit_h.pth"
python __init__.py --slides_path "D:/Develop/UBC/Datasets/R204brafv600e/Slides" --annotations_path "D:/Develop/UBC/Datasets/R204brafv600e/annotation" --output_path "D:/Develop/UBC/Datasets/R204brafv600e/Results_new_downsample_4" --slide_down_sample_rate 4 --overlay_down_sample_rate 4 --deepliif --model_dir "D:/Develop/UBC/DeepLIIF_Latest_Model" --tile_size 256 --mask_generator --model_path "D:/Develop/UBC/extract_areas/models/sam_vit_h.pth"
python __init__.py --slides_path "D:/Develop/UBC/Datasets/Breast_TMA/Slides" --output_path "D:/Develop/UBC/Datasets/Breast_TMA/debug" --slide_down_sample_rate 1 --overlay_down_sample_rate 4 --deepliif --model_dir "D:/Develop/UBC/DeepLIIF_Latest_Model" --tile_size 256 --mask_generator --model_path "D:/Develop/UBC/extract_areas/models/sam_vit_h.pth"

python __init__.py --slides_path "D:/Develop/UBC/Datasets/R204brafv600e/Slides" --annotations_path "D:/Develop/UBC/Datasets/R204brafv600e/annotation/BRAFV600E" --output_path "D:/Develop/UBC/Datasets/R204brafv600e/patches" --slide_down_sample_rate 1 --overlay_down_sample_rate 4 --deepliif --model_dir "D:/Develop/UBC/DeepLIIF_Latest_Model" --tile_size 256 --mask_generator --model_path "D:/Develop/UBC/extract_areas/models/sam_vit_h.pth" --cell_classifier --cell_classifier_model "D:/Develop/UBC/extract_areas/models/model_epoch_3_1.pth"
