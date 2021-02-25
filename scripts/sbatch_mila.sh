#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=150G
#SBATCH --time=3:00:00
#SBATCH -o /network/tmp1/<user>/slurm-%j.out  # Write the log on tmp1

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate <env_name>

# 3. Copy your dataset on the compute node
cp /network/data/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/