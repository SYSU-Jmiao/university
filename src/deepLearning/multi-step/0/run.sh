export BUCKET_NAME=yonidavidson-university
export JOB_NAME="multi_step_0_$(date +%Y%m%d_%H%M%S)"
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
    --data-location gs://$BUCKET_NAME/data/RML2016.10a_dict.dat
