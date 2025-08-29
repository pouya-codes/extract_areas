#!/bin/bash
#SBATCH --job-name=fastapi-job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4              # remove if no GPU needed
#SBATCH --time=19-10:00:00
#SBATCH --output=api_job.out
#SBATCH --error=api_job.err
#SBATCH -p gpu2080
#SBATCH -w dlhost03
#SBATCH --mem 6000

# Load your environment
# bash /projects/ovcare/users/pouya_ahmadvand/miniconda3/etc/profile.d/conda.sh
#source activate fastapi_env        # or conda env

# Get the compute node's hostname
HOST=$(hostname)
echo "API running on $HOST"
# Run FastAPI, binding to the node's hostname
python -m uvicorn app:app --host 127.0.0.1 --port 8001 --reload
