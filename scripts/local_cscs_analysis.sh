cd ..
python3 -m trainer.main \
    --batch_size 16 \
    --hidden 16 \
    --embed_dim 16 \
    --heads 1 \
    --depth 1 \
    --drop_prob 0.1 \
    --learning_rate 1e-2 \
    --epochs 1 \
    --model_type attention \
    --lr_scheduler plateau \
    --within_epoch_interval 5 \
    --patience 1 \
    --name_run first_trials \
    --split_method random \
    --manual_seed 42 \
    --num_workers 0 \
    --es_patience 3 \
    --state_dict_fname "saved_models/first_trial_100.pth" \
    --wt_seqs_file "escape_validation/anchor_seqs.fasta" \
    --POI_file "escape_validation/regions_of_interest.json" \
    --eval_batch_size 2 \
    --max_len 1425 \
    --min_len 1300 \
#    --cscs_debug \
#    --wandb
#    --truncate \