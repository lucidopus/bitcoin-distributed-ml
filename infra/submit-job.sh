


ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi


JOB_NAME=""
DATA_PERCENTAGE=""

show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --job-name NAME       Name of the job file (e.g., eda.py). Must be in 'jobs/' dir. (REQUIRED)"
    echo "  --cluster-name NAME   Name of the Dataproc cluster (Overrides CLUSTER_NAME from .env file)"
    echo "  --data-percentage NUM Percentage of data to use (Required for jobs in 'training/' folder)"
    echo "  --help                Show this help message"
    echo ""
}


while [[ "$
    case $1 in
        --job-name) JOB_NAME="$2"; shift ;;
        --cluster-name) CLUSTER_NAME="$2"; shift ;;
        --data-percentage) DATA_PERCENTAGE="$2"; shift ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done


if [ -z "$JOB_NAME" ]; then
    echo "Error: --job-name argument is required."
    show_help
    exit 1
fi

if [ -z "$CLUSTER_NAME" ]; then
    echo "Error: CLUSTER_NAME must be set via .env or --cluster-name argument."
    show_help
    exit 1
fi


if [[ "$JOB_NAME" == *"training/"* ]]; then
    if [ -z "$DATA_PERCENTAGE" ]; then
        echo "Error: --data-percentage argument is REQUIRED for jobs in the 'training/' folder."
        show_help
        exit 1
    fi
    echo "Training job detected. Using data percentage: $DATA_PERCENTAGE%"
fi


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


SPARK_PROPERTIES="spark.submit.deployMode=cluster"


if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env..."
    
    
    while IFS='=' read -r key value; do
        
        [[ "$key" =~ ^
        [[ -z "$key" ]] && continue
        
        
        value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
        
        
        [[ -z "$value" ]] && continue
        
        
        SPARK_PROPERTIES="${SPARK_PROPERTIES},spark.yarn.appMasterEnv.${key}=${value}"
        SPARK_PROPERTIES="${SPARK_PROPERTIES},spark.executorEnv.${key}=${value}"
        SPARK_PROPERTIES="${SPARK_PROPERTIES},spark.driverEnv.${key}=${value}"
        
        echo "  ✓ Added: $key"
    done < <(grep -v '^[[:space:]]*$' "$ENV_FILE")
fi


JOB_ARGS=""
if [ ! -z "$DATA_PERCENTAGE" ]; then
    JOB_ARGS="--data-percentage $DATA_PERCENTAGE"
fi


JOB_OUTPUT=$(gcloud dataproc jobs submit pyspark "$JOB_PATH" \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --properties="$SPARK_PROPERTIES" \
    --format="value(reference.jobId)" \
    -- $JOB_ARGS)

JOB_ID=$JOB_OUTPUT

echo ""
echo "Job ID: $JOB_ID"
echo "Waiting for job to complete..."


gcloud dataproc jobs wait "$JOB_ID" --region=$REGION --project=$PROJECT_ID

echo ""
echo "----------------------------------------------------------------"
echo "Fetching job logs..."
echo "----------------------------------------------------------------"


CLUSTER_UUID=$(gcloud dataproc clusters describe $CLUSTER_NAME --region=$REGION --format="value(clusterUuid)")


LOG_PATH="gs://${BUCKET_NAME}/google-cloud-dataproc-metainfo/${CLUSTER_UUID}/jobs/${JOB_ID}/driveroutput.000000001"

echo "Checking log path: $LOG_PATH"
if gsutil -q stat "$LOG_PATH"; then
    echo ""
    echo "=== JOB OUTPUT ==="
    gsutil cat "$LOG_PATH"
else
    echo "Warning: Driver output not found at expected location"
    echo "Check logs at: https://console.cloud.google.com/dataproc/jobs/${JOB_ID}?project=${PROJECT_ID}&region=${REGION}"
fi
