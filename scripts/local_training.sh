cd ..
python3 -m trainer.main \
    --batch_size 16 \
    --hidden 16 \
    --embed_dim 16 \
    --max_len 1425 \
    --min_len 1300 \
    --heads 1 \
    --depth 1 \
    --drop_prob 0.1 \
    --learning_rate 1e-2 \
    --epochs 10 \
    --model_type "bilstm" \
    --lr_scheduler plateau \
    --within_epoch_interval 5 \
    --patience 1 \
    --name_run first_trial \
    --split_method random \
    --manual_seed 42 \
    --num_workers 0 \
    --es_patience 3 \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --cscs_debug \
#    --wandb
