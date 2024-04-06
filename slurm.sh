#!/bin/bash

#SBATCH --job-name=SemanticSam
#SBATCH --account=ml20

#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:V100:2
#SBATCH --partition=m3g

#SBATCH --mem-per-cpu=60000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=6-00:00:00

# Set the file for output (stdout)
#SBATCH --output=/home/msuleman/ml20_scratch/fyp_galaxy/SemanticGalSAM/logs/test.out

# Set the file for error log (stderr)
#SBATCH --error=/home/msuleman/ml20_scratch/fyp_galaxy/SemanticGalSAM/logs/test.err

# Command to run a gpu job
# For example:
source ~/ed74_scratch/msuleman/miniconda/bin/activate
conda activate fyp_dinov2
cd ~/ml20_scratch/fyp_galaxy/SemanticGalSAM

python train.py --epochs 10 --lr 5e-4 --save_path "saves/test.ckpt" --batch_size 10
