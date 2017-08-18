export BUCKET_NAME=yonidavidson-university
export JOB_NAME="radioMl_cwt_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1


gcloud ml-engine local train \
  --job-dir $JOB_DIR \
  --module-name trainer.radioML_cwt \
  --package-path ./trainer \
  -- \
  --train-file ./data/mnist.pkl


  # cloud ml-engine jobs submit training $JOB_NAME \
  #   --job-dir $JOB_DIR \
  #   --runtime-version 1.0 \
  #   --module-name trainer.mnist_mlp \
  #   --package-path ./trainer \
  #   --region $REGION \
  #   -- \
  #   --train-file gs://$BUCKET_NAME/data/mnist.pkl