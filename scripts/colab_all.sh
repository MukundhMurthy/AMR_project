cd ..
python3 -m trainer.main \
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
    --model_type bilstm \
    --lr_scheduler plateau \
    --within_epoch_interval 5 \
    --patience 2 \
    --name_run fourth_trial_save_weights \
    --split_method random \
    --manual_seed 42 \
    --num_workers 1 \
    --es_patience 4 \
    --POI_file "escape_validation/regions_of_interest.json" \
    --wt_seqs_file "escape_validation/anchor_seqs.fasta" \
    --eval_batch_size 4 \
    --file_column_dictionary "escape_validation/file_column_dict.json" \
    --scaling "min_max" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --state_dict_fname "saved_models/fourth_trial_esm_embed.pth" \
    --train \
    --cscs_debug \
#    --benchmark \
#    --cscs \
#    --analyze_embs \
#    --calc_metrics \
#    --emb_is_prob_vec \
#    --wandb \
#    --cscs_debug \
#    --benchmark \
#    --truncate \
