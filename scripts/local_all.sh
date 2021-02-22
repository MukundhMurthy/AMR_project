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
    --epochs 70 \
    --model_type attention \
    --lr_scheduler plateau \
    --within_epoch_interval 5 \
    --patience 2 \
    --name_run third_trial \
    --split_method random \
    --manual_seed 42 \
    --num_workers 1 \
    --es_patience 4 \
    --state_dict_fname "saved_models/compressed_comb.pth" \
    --POI_file "escape_validation/regions_of_interest.json" \
    --wt_seqs_file "escape_validation/anchor_seqs.fasta" \
    --eval_batch_size 16 \
    --file_column_dictionary "escape_validation/file_column_dictionary" \
    --scaling "min_max" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --embedding_fname "saved_models/compressed_comb.pth" \
    --analyze_embs \
#    --cscs_debug \
#    --train \
#    --calc_metrics \
#    --benchmark \
#    --truncate \
#    --wandb