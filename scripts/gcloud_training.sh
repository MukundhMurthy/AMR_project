
PACKAGE_PATH='../trainer'
MODULE_NAME='trainer.transformer'
JOB_DIR='gs://amr-transformer'
REGION='us-east1'
IMAGE_URI='gcr.io/cloud-ml-public/training/pytorch-gpu.1-6'
JOB_NAME=pytorch_job_$(date +%Y%m%d_%H%M%S)

gcloud ai-platform jobs submit training $JOB_NAME \
    --job-dir=$JOB_DIR  \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier PREMIUM_1 \
    --package-path=$PACKAGE_PATH \
    --module-name=$MODULE_NAME \
    --region=$REGION \
    -- \
    --batch_size 16 \
    --hidden 512 \
    --embed_dim 32 \
    --heads 8 \
    --depth 3 \
    --drop_prob 0.1 \
    --learning_rate 0.001 \
    --epochs 5 \
    --model_type attention \
    --lr_scheduler plateau \
    --within_epoch_interval 2 \
    --patience 1 \
    --name_run first_trial \
    --split_method random \
    --manual_seed 42 \
    --num_workers 1 \
    --wandb



gcloud ai-platform jobs stream-logs ${JOB_NAME}

#    --staging-bucket=$STAGING_BUCKET \
#STAGING_BUCKET='gs://amr-transformer/trial'