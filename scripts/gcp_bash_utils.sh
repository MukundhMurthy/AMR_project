#!/bin/bash

PORT=8081
VM='deeplearning-vm-5'

initialize_vm () {
  gcloud compute ssh $VM \
    --project amr-proj \
    --zone us-central1-a \
    -- -L ${PORT}:localhost:${PORT}
  pip install jupyter_http_over_ws
  jupyter serverextension enable --py jupyter_http_over_ws
  mkdir AMR_transformer
  jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=${PORT} \
    --NotebookApp.port_retries=0
}

transfer_trainer_scripts () {
  gcloud compute copy-files deeplearning-5-vm:~/AMR_transformer/figures/umaprpoB_louvain.png  \
  ~/AMR_transformer/figures --zone=us-east1-c
}

transfer_trainer_scripts