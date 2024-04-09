#!/bin/bash

# Define other variables
lr=5e-4
epochs=5
lora_rank=4
lora_alpha=1
batch_size=3

job_name="lr_${lr}_epochs_${epochs}_lora_rank_${lora_rank}_lora_alpha_${lora_alpha}_batch_size_${batch_size}"

output_file="/home/msuleman/ml20_scratch/fyp_galaxy/SemanticGalSAM/logs/${job_name}.out"
error_file="/home/msuleman/ml20_scratch/fyp_galaxy/SemanticGalSAM/logs/${job_name}.err"

# Generate SLURM submission script dynamically
cat <<EOT > submission_script.sh
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --account=ml20

#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:V100:2
#SBATCH --partition=m3g

#SBATCH --mem-per-cpu=60000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=6-00:00:00

# Set the file for output (stdout)
#SBATCH --output=$output_file

# Set the file for error log (stderr)
#SBATCH --error=$error_file

# Command to run a gpu job
# For example:
source ~/ed74_scratch/msuleman/miniconda/bin/activate
conda activate fyp_dinov2
cd ~/ml20_scratch/fyp_galaxy/SemanticGalSAM

python train.py --epochs $epochs --lr $lr --lora_rank $lora_rank --lora_alpha $lora_alpha --batch_size $batch_size
EOT

# Submit the script to SLURM
sbatch submission_script.sh

rm submission_script.sh
