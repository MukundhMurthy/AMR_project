cd ..
python3 -m trainer.cscs \
    --batch_size 32 \
    --hidden 512 \
    --embed_dim 32 \
    --heads 4 \
    --depth 3 \
    --drop_prob 0.1