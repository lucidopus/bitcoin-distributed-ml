CLUSTER_NAME="bitcoin-cluster-dev"
REGION="us-central1"
PROJECT_ID="bitcoin-trend-prediction1"

echo "Deleting cluster: $CLUSTER_NAME in $PROJECT_ID..."

gcloud dataproc clusters delete $CLUSTER_NAME \
    --region $REGION \
    --project $PROJECT_ID \
    --quiet

if [ $? -eq 0 ]; then
    echo "Cluster deleted successfully."
else
    echo "Cluster deletion FAILED."
    exit 1
fi
