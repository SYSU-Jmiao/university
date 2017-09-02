# export BUCKET_NAME=yonidavidson-university
# export JOB_NAME="radioMl_cwt_$(date +%Y%m%d_%H%M%S)"
# export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
# export REGION=us-east1

# local
# gcloud ml-engine local train \
#   --job-dir $JOB_DIR \
#   --module-name trainer.radioML_cwt \
#   --package-path ./trainer \
#   -- \
#   --train-file ./data/RML2016.10a_dict.dat

#cloud
  # gcloud ml-engine jobs submit training $JOB_NAME \
  #   --job-dir $JOB_DIR \
  #   --runtime-version 1.0 \
  #   --module-name trainer.radioML_cwt \
  #   --package-path ./trainer \
  #   --region $REGION \
  #   -- \
  #   --trpreprocess_cwt_ain-file gs://$BUCKET_NAME/data/RML2016.10a_dict.dat


# export BUCKET_NAME=yonidavidson-university
# export JOB_NAME="preprocess_cwt_$(date +%Y%m%d_%H%M%S)"
# export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
# export REGION=us-east1

#     #cloud - preprocess
#   gcloud ml-engine jobs submit training $JOB_NAME \
#     --job-dir $JOB_DIR \
#     --runtime-version 1.0 \
#     --module-name trainer.preprocess_cwt \
#     --package-path ./trainer \
#     --region $REGION \
#     -- \
#     --train-file gs://$BUCKET_NAME/data/RML2016.10a_dict.dat

#     # --scale-tier=STANDARD_1 \


# local preprocess

export BUCKET_NAME=yonidavidson-university
export JOB_NAME="preprocess_cwt_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1

    #cloud - preprocess
gcloud ml-engine local train \
  --job-dir $JOB_DIR \
  --module-name trainer.preprocess_cwt \
  --package-path ./trainer \
  -- \
  --train-file ./data/RML2016.10a_dict.dat