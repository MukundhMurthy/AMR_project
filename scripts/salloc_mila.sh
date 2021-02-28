#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=150G
#SBATCH --time=3:00:00
#SBATCH -o /network/tmp1/<user>/slurm-%j.out  # Write the log on tmp1

# 1. Load the required modules
module --quiet load python/3.7

# 2. Load your environment
source $HOME/amr_transformer/bin/activate

# 3. Copy your dataset on the compute node

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python3 -m main.py \
    --batch_size 32 \
    --hidden 512 \
    --embed_dim 32 \
    --max_len 1425 \
    --min_len 1300 \
    --heads 4 \
    --depth 3 \
    --drop_prob 0.1 \
    --learning_rate 0.001 \
    --epochs 60 \
    --model_type attention \
    --lr_scheduler plateau \
    --within_epoch_interval 5 \
    --patience 2 \
    --name_run fourth_trial \
    --split_method random \
    --manual_seed 42 \
    --num_workers 1 \
    --es_patience 4 \
    --POI_file "escape_validation/regions_of_interest.json" \
    --wt_seqs_file "anchor_seqs.fasta" \
    --eval_batch_size 16 \
    --file_column_dictionary "escape_validation/file_column_dict.json" \
    --scaling "min_max" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --state_dict_fname "saved_models/fourth_trial_50.pth" \
    --analyze_embs \
    --calc_metrics \
    --wandb

# 5. Copy whatever you want to save on $SCRATCH
