# bitcoin-distributed-ml

**Distributed Bitcoin Trend Prediction using Apache Spark on Google Cloud Dataproc**

This project demonstrates a scalable Machine Learning pipeline for predicting short-term Bitcoin price movements. It leverages **Apache Spark (PySpark)** for distributed data processing and model training, orchestrated on a **Google Cloud Dataproc** cluster.

## 🏗 Architecture

The pipeline follows a standard ETL-Model workflow:

1.  **Ingestion**: Fetches raw Bitcoin historical data (OpenML) and stores it in Google Cloud Storage (GCS) - *Bronze Layer*.
2.  **Preprocessing (Spark)**:
    *   Loads data from GCS.
    *   Handles missing values via time-series based imputation (Backfill/Forward-fill).
    *   Performs feature engineering (Moving Averages, Volatility, Target generation).
    *   Vectorizes features for MLLib.
3.  **Model Training**: Trains distributed ML models (Logistic Regression, Random Forest, GBT) to predict price direction.

## 📂 Project Structure

```text
bitcoin-distributed-ml/
├── docs/
│   └── PROJECT_PLAN.md       # Detailed architectural roadmap
├── infra/
│   ├── create_cluster.sh     # Script to provision the Dataproc cluster
│   └── delete_cluster.sh     # Script to tear down the cluster
├── jobs/
│   ├── data-preprocessing.py # Spark job for cleaning and vectorization
│   └── feature-engineering.py# Feature generation logic
├── README.md                 # This file
└── requirements.txt          # Local python dependencies
```

## 🚀 Getting Started

### Prerequisites

*   **Google Cloud SDK (`gcloud`)**: Installed and authenticated (`gcloud auth login`).
*   **Python 3.9+**: For local development.
*   **GCP Project**: A valid Google Cloud project with Dataproc and GCS APIs enabled.

### 1. Infrastructure Setup

Provision a Dataproc cluster using the provided script. This sets up a master node and workers with the necessary initialization actions.

```bash
cd infra
./create_cluster.sh
```

*Note: Ensure you have configured your project ID and region variables if they differ from the defaults in the script.*

### 2. Running Spark Jobs

Submit jobs to the cluster using the `gcloud dataproc jobs submit` command.

**Example: Feature Engineering**

```bash
gcloud dataproc jobs submit pyspark jobs/feature-engineering.py \
    --cluster=bitcoin-cluster \
    --region=us-central1 \
    --jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar
```
