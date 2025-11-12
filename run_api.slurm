#!/bin/bash
#SBATCH --job-name=fastapi-job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1              # remove if no GPU needed
#SBATCH --time=19-10:00:00
#SBATCH --output=api_job.out
#SBATCH --error=api_job.err
#SBATCH -p gpu3090
# SBATCH -w dlhost03
#SBATCH --mem 4000

source /projects/ovcare/users/pouya_ahmadvand/miniconda3/etc/profile.d/conda.sh
conda activate fastapi_env        

cd /projects/ovcare/users/pouya_ahmadvand/develop/API/extract_areas

# Get the compute node's hostname
HOST=$(hostname)
echo "API running on $HOST"
# Run FastAPI, binding to the node's hostname
python app_refactored.py 