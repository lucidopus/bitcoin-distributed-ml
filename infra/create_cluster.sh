CLUSTER_NAME="bitcoin-cluster-dev"
REGION="us-central1"
ZONE="us-central1-a"
PROJECT_ID="bitcoin-trend-prediction1"
BUCKET_NAME="bitcoin-trend-prediction1-data"

echo "Creating cluster: $CLUSTER_NAME in $PROJECT_ID..."

gcloud dataproc clusters create $CLUSTER_NAME \
    --region $REGION \
    --zone $ZONE \
    --master-machine-type n1-standard-2 \
    --master-boot-disk-size 50GB \
    --num-workers 2 \
    --worker-machine-type n1-standard-2 \
    --worker-boot-disk-size 50GB \
    --image-version 2.1-debian11 \
    --enable-component-gateway \
    --project $PROJECT_ID \
    --bucket $BUCKET_NAME \
    --scopes 'https://www.googleapis.com/auth/cloud-platform'

if [ $? -eq 0 ]; then
    echo "Cluster created successfully."
else
    echo "Cluster creation FAILED."
    exit 1
fi


# gcloud dataproc jobs submit pyspark gs://bitcoin-trend-prediction1-data/feature-engineering/feature-engineering.py \
#     --cluster=bitcoin-cluster-dev \
#     --region=us-central1