# Bitcoin Trend Prediction: Distributed ML Pipeline Plan

## Phase 1: Infrastructure Setup (GCP & Dataproc)
**Goal:** Create the "playground" where the Big Data processing happens.

1.  **GCP Project Setup:**
    * Create a new project in Google Cloud Platform (e.g., `bia678-bitcoin-group-b`).
    * Enable the following APIs:
        * **Dataproc API** (For the cluster)
        * **Compute Engine API** (For the VMs)
        * **Cloud Storage API** (For storing data)

2.  **Storage Setup (Data Lake):**
    * Create a Google Cloud Storage (GCS) Bucket: `gs://bia678-group-b-data/`
    * Create folders: `/raw`, `/processed`, `/models`, `/logs`.

3.  **Cluster Creation (The "Scale Out" Baseline):**
    * We will start with a small cluster (1 Master, 2 Workers) for development.
    * **Command:**
        ```bash
        gcloud dataproc clusters create bitcoin-cluster-dev \
            --region us-central1 \
            --zone us-central1-a \
            --master-machine-type n1-standard-4 \
            --worker-machine-type n1-standard-4 \
            --num-workers 2 \
            --image-version 2.0-debian10 \
            --optional-components JUPYTER,ANACONDA \
            --enable-component-gateway
        ```
    * *Note: This installs Spark, Python, and Jupyter automatically.*

---

## Phase 2: Data Ingestion & Engineering
**Goal:** Get the OpenML data and CVIX data into the same format in GCS.

1.  **Fetch Bitcoin Data (OpenML):**
    * **Action:** Instead of downloading locally, open a Jupyter Notebook on the Dataproc Master Node.
    * **Code:** Use `sklearn.datasets.fetch_openml` (ID: 43939 or search "cryptocurrency") to fetch the dataframe.
    * **Save:** Write this dataframe immediately to GCS as a Parquet file (much faster for Spark than CSV).
        ```python
        # Pandas on Master Node
        df.to_parquet('gs://bia678-group-b-data/raw/bitcoin_openml.parquet')
        ```

2.  **Fetch CVIX (Volatility) Data:**
    * **Action:** Since CVIX doesn't have a clean API, download the historical CSV manually from [Investing.com](https://www.investing.com/indices/crypto-volatility-index-historical-data) or [CVI.finance](https://cvi.finance/).
    * **Upload:** Manually upload this CSV to `gs://bia678-group-b-data/raw/cvix_data.csv`.

3.  **ETL Pipeline (Spark Job):**
    * **Challenge:** Bitcoin data is **minute-level**; CVIX is **daily/hourly**.
    * **Logic:** You must "Forward Fill" the CVIX data.
    * **Spark Logic:**
        1.  Load Bitcoin Parquet -> Spark DataFrame.
        2.  Load CVIX CSV -> Spark DataFrame.
        3.  Join on `Date`.
        4.  Calculate the SMA-15 feature: `((Close - SMA15) / SMA15) * 100`.
        5.  Create Target Label: `1` if Future Close > Current Close, else `0`.
        6.  Save final table to `gs://bia678-group-b-data/processed/training_data`.

---

## Phase 3: Model Development (The "Code")
**Goal:** Write the PySpark scripts for the 3 algorithms.

1.  **Random Forest Classifier (Spark MLlib):**
    * Standard implementation. Good baseline.
    * *Key Param:* `numTrees=100`.

2.  **XGBoost Classifier (Spark Wrapper):**
    * **Critical:** You need the `xgboost4j-spark` or `xgboost-pyspark` library.
    * *Action:* When creating the cluster, you may need to add initialization actions to install `xgboost`.
    * Alternatively, simple `pip install xgboost` on the cluster nodes if using the Python wrapper.

3.  **Multilayer Perceptron (MLP):**
    * Use `pyspark.ml.classification.MultilayerPerceptronClassifier`.
    * *Layers:* `[input_size, 64, 32, 2]` (Input -> Hidden -> Hidden -> Output).

---

## Phase 4: The "Scale Out" Experiment (For the Professor)
**Goal:** Generate the metrics that prove you did "Big Data" engineering.

1.  **Experiment A (2 Workers):**
    * Run the training job for Random Forest on the 2-worker cluster.
    * **Record:** Total Training Time (e.g., 14 minutes). Accuracy (e.g., 54%).

2.  **Experiment B (Scaling Up Data):**
    * Run the same job but use only 50% of the data, then 100%.
    * **Record:** Time difference.

3.  **Experiment C (Scaling Out Hardware):**
    * **Action:** Resize the cluster (Dynamic Scaling).
        ```bash
        gcloud dataproc clusters update bitcoin-cluster-dev --num-workers 4
        ```
    * Run the training job again.
    * **Hypothesis:** Training time should drop (e.g., to 8 minutes).
    * *This specific comparison is what will get you the "A".*

---

## Phase 5: Final Output & Reporting
**Goal:** Package it up.

1.  **The Artifacts:**
    * Source Code (Python scripts).
    * Saved Models (in GCS).
    * `results.csv`: A table containing [Model, Nodes, Rows, Training_Time, F1_Score].

2.  **The Presentation:**
    * Show the **Infrastructure Diagram** (GCS -> Dataproc -> Model).
    * Show the **Scalability Chart** (Bar chart: Time vs. Number of Nodes).
    * Show the **CVIX Impact** (Did adding volatility improve accuracy?).