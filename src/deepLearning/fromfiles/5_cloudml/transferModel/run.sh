export BUCKET_NAME=yonidavidson-university
export JOB_NAME="tranfer_5_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1
gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version 1.0 \
    --module-name trainer.task \
    --package-path ./trainer \
    --scale-tier basic-gpu \
    --region $REGION \
    -- \
    --data-location gs://$BUCKET_NAME/data/cwt_i_over_q_clean
