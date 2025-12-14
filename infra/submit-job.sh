# Load environment variables
if [ -f "$(dirname "$0")/../.env" ]; then
    source "$(dirname "$0")/../.env"
fi

# Default values
JOB_NAME=""

show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --job-name NAME    Name of the job file (e.g., eda.py). Must be in 'jobs/' dir. (REQUIRED)"
    echo "  --help             Show this help message"
    echo ""
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --job-name) JOB_NAME="$2"; shift ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$JOB_NAME" ]; then
    echo "Error: --job-name argument is required."
    show_help
    exit 1
fi

# Construct local job path
JOB_PATH="$(dirname "$0")/../jobs/$JOB_NAME"

if [ ! -f "$JOB_PATH" ]; then
    echo "Error: Job file not found at $JOB_PATH"
    exit 1
fi

echo "----------------------------------------------------------------"
echo "Submitting Job: $JOB_PATH"
echo "Target Cluster: $CLUSTER_NAME ($REGION)"
echo "Bucket Context: $BUCKET_NAME"
echo "----------------------------------------------------------------"

gcloud dataproc jobs submit pyspark "$JOB_PATH" \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --properties="spark.submit.deployMode=cluster,spark.yarn.appMasterEnv.BUCKET_NAME=${BUCKET_NAME},spark.executorEnv.BUCKET_NAME=${BUCKET_NAME}"
