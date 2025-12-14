# Load environment variables
if [ -f "$(dirname "$0")/../.env" ]; then
    source "$(dirname "$0")/../.env"
fi

# Default values
NUM_WORKERS=""

show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --workers N        Number of worker nodes (REQUIRED)"
    echo "  --help             Show this help message"
    echo ""
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --workers) NUM_WORKERS="$2"; shift ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$NUM_WORKERS" ]; then
    echo "Error: --workers argument is required."
    show_help
    exit 1
fi

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
