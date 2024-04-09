#!/bin/bash

#SBATCH --job-name=SemanticSAM
#SBATCH --account=ml20

#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:V100:2
#SBATCH --partition=m3g

#SBATCH --mem-per-cpu=60000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=6-00:00:00

# Set the file for output (stdout)
#SBATCH --output="/home/msuleman/ml20_scratch/fyp_galaxy/SemanticGalSAM/logs/baseline_maskdecoder.out"

# Set the file for error log (stderr)
#SBATCH --error="/home/msuleman/ml20_scratch/fyp_galaxy/SemanticGalSAM/logs/baseline_maskdecoder.err"

# Command to run a gpu job
# For example:
source ~/ed74_scratch/msuleman/miniconda/bin/activate
conda activate fyp_dinov2
cd ~/ml20_scratch/fyp_galaxy/SemanticGalSAM

python train.py --epochs 20 --lr 5e-4 --lora_rank 4 --lora_alpha 8 --batch_size 10 --wandb 1