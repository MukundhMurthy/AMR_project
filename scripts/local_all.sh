cd ..
python3 -m trainer.main \
    --batch_size 20 \
    --hidden 16 \
    --embed_dim 16 \
    --max_len 1425 \
    --min_len 1300 \
    --heads 1 \
    --depth 1 \
    --drop_prob 0.1 \
    --learning_rate 0.001 \
    --epochs 1 \
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
    --wt_seqs_file "escape_validation/anchor_seqs.fasta" \
    --eval_batch_size 16 \
    --file_column_dictionary "escape_validation/file_column_dict.json" \
    --state_dict_fname "saved_models/compressed_comb.pth" \
    --scaling "min_max" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --analyze_embs \
    --train \
    --calc_metrics \
    --cscs_debug \



#    --state_dict_fname "saved_models/compressed_comb.pth" \
#    --POI_file "escape_validation/regions_of_interest.json" \
#    --wt_seqs_file "escape_validation/anchor_seqs.fasta" \
#
#    --file_column_dictionary "escape_validation/file_column_dictionary" \
#
#    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
#    --embedding_fname "saved_models/compressed_comb.pth" \

#    --cscs_debug \
#    --train \
#    --calc_metrics \
#    --benchmark \
#    --truncate \
#    --wandb