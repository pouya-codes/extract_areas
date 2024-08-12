#!/bin/bash
#SBATCH --job-name ExtractAreas
#SBATCH --cpus-per-task 1
#SBATCH --chdir /projects/ovcare/classification/pouya/QA-QC/Codes/extract_areas
#SBATCH --output /projects/ovcare/classification/pouya/QA-QC/Codes/extract_areas/temp.out
#SBATCH --error /projects/ovcare/classification/pouya/QA-QC/Codes/extract_areas/temp.err
#SBATCH -p gpu3090,gpu3090short,dgxV100,rtx5000,gpuA6000
#SBATCH --time=1:00:00
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
singularity --version
# singularity run --bind /projects:/projects --nv my_container.sif python __init__.py
# singularity run singularity.sif python --version
# singularity run --bind /projects:/projects --nv singularity.sif python __init__.py --slides_path "/projects/ovcare/classification/pouya/QA-QC/datasets/R204brafv600e/Slides" --output_path "/projects/ovcare/classification/pouya/QA-QC/datasets/R204brafv600e/final" --qupath_dearray_paths "/projects/ovcare/classification/pouya/QA-QC/datasets/R204brafv600e/Dearray"  --slide_down_sample_rate 1 --overlay_down_sample_rate 4 --deepliif --model_dir "/projects/ovcare/classification/pouya/QA-QC/Models/DeepLIIF_Latest_Model" --tile_size 256 --mask_generator --model_path "/projects/ovcare/classification/pouya/QA-QC/Models/sam_vit_h.pth" --cell_classifier --cell_classifier_model "/projects/ovcare/classification/pouya/QA-QC/Models/model_epoch_3_1.pth" --staining "cytoplasm" --metadata "./CPQA.xlsx" --metadata_sheet "BRAFV600E assessments"

singularity run --bind /projects:/projects --nv singularity.sif python __init__.py --slides_path "/projects/ovcare/classification/pouya/QA-QC/datasets/TNP_Array/Slides" --output_path "/projects/ovcare/classification/pouya/QA-QC/datasets/TNP_Array/final" --slide_down_sample_rate 1 --overlay_down_sample_rate 4 --deepliif --model_dir "/projects/ovcare/classification/pouya/QA-QC/Models/DeepLIIF_Latest_Model" --tile_size 256 --mask_generator --model_path "/projects/ovcare/classification/pouya/QA-QC/Models/sam_vit_h.pth" --staining "nuclear" --metadata "./CPQA.xlsx" --metadata_sheet "ER assessments"
