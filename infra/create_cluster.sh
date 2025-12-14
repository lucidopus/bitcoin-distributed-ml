# Load environment variables
if [ -f "$(dirname "$0")/../.env" ]; then
    source "$(dirname "$0")/../.env"
fi

NUM_WORKERS=2

echo "Creating cluster: $CLUSTER_NAME in $PROJECT_ID..."

gcloud dataproc clusters create $CLUSTER_NAME \
    --region $REGION \
    --zone $ZONE \
    --master-machine-type n1-standard-2 \
    --master-boot-disk-size 50GB \
    --num-workers $NUM_WORKERS \
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
