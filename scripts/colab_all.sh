#!/bin/sh

# shellcheck disable=SC2162

cd ..

esm () {
  python3 -m trainer.main \
    --embed_dim 32 \
    --epochs 60 \
    --model_type esm \
    --name_run $1 \
    --eval_batch_size 4 \
    --file_column_dictionary "escape_validation/file_column_dict.json" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --state_dict_fname "saved_models/fourth_trial_50_embed.pth" \
    --calc_metrics \
    --cscs \
    --wandb
}

embed_var () {
  python3 -m trainer.main \
    --embed_dim $1 \
    --epochs 60 \
    --model_type attention \
    --name_run $2 \
    --eval_batch_size 4 \
    --file_column_dictionary "escape_validation/file_column_dict.json" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --state_dict_fname "saved_models/fourth_trial_50_embed.pth" \
    --calc_metrics \
    --train \
    --cscs \
    --analyze_embs \
    --wandb \
}

Lk () {
    python3 -m trainer.main \
    --embed_dim 32 \
    --epochs 60 \
    --model_type attention \
    --name_run $2 \
    --eval_batch_size 4 \
    --file_column_dictionary "escape_validation/file_column_dict.json" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --state_dict_fname "saved_models/fourth_trial_50_embed.pth" \
    --L_k $1 \
    --calc_metrics \
    --train \
    --benchmark \
    --cscs \
    --analyze_embs \
    --wandb \
}

bilstm () {
    python3 -m trainer.main \
    --embed_dim $1 \
    --epochs 60 \
    --model_type bilstm \
    --name_run $2 \
    --eval_batch_size 4 \
    --file_column_dictionary "escape_validation/file_column_dict.json" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --state_dict_fname "saved_models/fourth_trial_50_embed.pth" \
    --calc_metrics \
    --train \
    --benchmark \
    --cscs \
    --analyze_embs \
    --wandb \
}

if [ "$1" = 'esm' ]
then
  esm $2
elif [ "$1" = 'embed_var' ]
then
  embed_var $2 $3
elif [ "$1" = 'Lk' ]
then
  Lk $2 $3
elif [ "$1" = 'bilstm' ]
then
  bilstm $2 $3
fi