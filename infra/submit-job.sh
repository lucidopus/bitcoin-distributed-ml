#!/bin/bash

# Load environment variables
ENV_FILE="$(dirname "$0")/../.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

# Default values
JOB_NAME=""

show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --job-name NAME       Name of the job file (e.g., eda.py). Must be in 'jobs/' dir. (REQUIRED)"
    echo "  --cluster-name NAME   Name of the Dataproc cluster (Overrides CLUSTER_NAME from .env file)"
    echo "  --help                Show this help message"
    echo ""
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --job-name) JOB_NAME="$2"; shift ;;
        --cluster-name) CLUSTER_NAME="$2"; shift ;;
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

if [ -z "$CLUSTER_NAME" ]; then
    echo "Error: CLUSTER_NAME must be set via .env or --cluster-name argument."
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

# Build properties string - use CLUSTER mode so env vars work
SPARK_PROPERTIES="spark.submit.deployMode=cluster"

# Automatically read all variables from .env and pass them to Spark
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env..."
    
    # Read each line from .env, skip comments and empty lines
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Remove quotes from value if present
        value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
        
        # Skip if value is empty
        [[ -z "$value" ]] && continue
        
        # Add to Spark properties for all components
        SPARK_PROPERTIES="${SPARK_PROPERTIES},spark.yarn.appMasterEnv.${key}=${value}"
        SPARK_PROPERTIES="${SPARK_PROPERTIES},spark.executorEnv.${key}=${value}"
        SPARK_PROPERTIES="${SPARK_PROPERTIES},spark.driverEnv.${key}=${value}"
        
        echo "  ✓ Added: $key"
    done < <(grep -v '^[[:space:]]*$' "$ENV_FILE")
fi

# Submit the job with all properties
JOB_OUTPUT=$(gcloud dataproc jobs submit pyspark "$JOB_PATH" \
    --cluster=$CLUSTER_NAME \
    --region=$REGION \
    --properties="$SPARK_PROPERTIES" \
    --format="value(reference.jobId)")

JOB_ID=$JOB_OUTPUT

echo ""
echo "Job ID: $JOB_ID"
echo "Waiting for job to complete..."

# Wait for job to finish
gcloud dataproc jobs wait "$JOB_ID" --region=$REGION --project=$PROJECT_ID

echo ""
echo "----------------------------------------------------------------"
echo "Fetching job logs..."
echo "----------------------------------------------------------------"

# Get the cluster UUID for log path
CLUSTER_UUID=$(gcloud dataproc clusters describe $CLUSTER_NAME --region=$REGION --format="value(clusterUuid)")

# Try to fetch driver output
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
