PACKAGE_PATH='../trainer'
MODULE_NAME='trainer.main'
JOB_DIR='gs://amr-transformer'
REGION='us-east1'
IMAGE_URI='gcr.io/cloud-ml-public/training/pytorch-gpu.1-6'
JOB_NAME=pytorch_job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir=$JOB_DIR  \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier custom \
    --master-machine-type complex_model_l_gpu \
    --package-path=$PACKAGE_PATH \
    --module-name=$MODULE_NAME \
    --region=$REGION \
    -- \
    --batch_size 32 \
    --hidden 512 \
    --embed_dim 32 \
    --heads 4 \
    --depth 3 \
    --drop_prob 0.1 \
    --learning_rate 0.001 \
    --epochs 70 \
    --model_type attention \
    --lr_scheduler plateau \
    --within_epoch_interval 5 \
    --patience 2 \
    --name_run second_trial \
    --split_method random \
    --manual_seed 42 \
    --num_workers 1 \
    --es_patience 4 \
    --wt_seqs_file "anchor_seqs.fasta" \
    --POI_file "regions_of_interest.json" \
    --eval_batch_size 16 \
    --max_len 1425 \
    --min_len 1300 \
    --eval_batch_size 32 \
    --wandb \
#    --truncate \
#    --cscs_debug