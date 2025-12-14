# Load environment variables
if [ -f "$(dirname "$0")/../.env" ]; then
    source "$(dirname "$0")/../.env"
fi

# Default values
CLUSTER_NAME=""

show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cluster-name NAME Name of the cluster to delete (REQUIRED)"
    echo "  --help             Show this help message"
    echo ""
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cluster-name) CLUSTER_NAME="$2"; shift ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

# Validate required arguments
if [ -z "$CLUSTER_NAME" ]; then
    echo "Error: --cluster-name argument is required."
    show_help
    exit 1
fi

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
