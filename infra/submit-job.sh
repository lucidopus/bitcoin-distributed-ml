# Load environment variables
if [ -f "$(dirname "$0")/../.env" ]; then
    source "$(dirname "$0")/../.env"
fi

JOB="gs://${BUCKET_NAME}/feature-engineering/feature-engineering.py"

echo "----------------------------------------------------------------"
echo "Submitting Job: $JOB"
echo "Target Cluster: $CLUSTER_NAME ($REGION)"
echo "Bucket Context: $BUCKET_NAME"
echo "----------------------------------------------------------------"

gcloud dataproc jobs submit pyspark $JOB \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --properties="spark.yarn.appMasterEnv.BUCKET_NAME=${BUCKET_NAME},spark.executorEnv.BUCKET_NAME=${BUCKET_NAME}"
