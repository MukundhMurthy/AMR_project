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
    --master-machine-type complex_model_m_gpu \
    --package-path=$PACKAGE_PATH \
    --module-name=$MODULE_NAME \
    --region=$REGION \
    -- \
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
    --model_type attention \
    --lr_scheduler plateau \
    --within_epoch_interval 5 \
    --patience 2 \
    --name_run fourth_trial \
    --split_method random \
    --manual_seed 42 \
    --num_workers 1 \
    --es_patience 4 \
    --POI_file "regions_of_interest.json" \
    --wt_seqs_file "anchor_seqs.fasta" \
    --eval_batch_size 16 \
    --file_column_dictionary "file_column_dictionary" \
    --scaling "min_max" \
    --uniprot_seqs_fname "uniprot_gpb_rpob.fasta" \
    --analyze_embs \
    --train \
    --calc_metrics \
    --wandb
#    --cscs_debug \
#    --benchmark \
#    --truncate \
