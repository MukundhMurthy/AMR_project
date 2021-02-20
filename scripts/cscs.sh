cd ..
python3 -m trainer.cscs \
    --batch_size 32 \
    --hidden 512 \
    --embed_dim 32 \
    --heads 4 \
    --depth 3 \
    --drop_prob 0.1 \
    --max_len 1425 \
    --min_len 1300 \
    --state_dict_fname "saved_models/compressed_comb.pth" \
    --wt_seqs_file "escape_validation/anchor_seqs.fasta" \
    --POI_file "escape_validation/regions_of_interest.json"