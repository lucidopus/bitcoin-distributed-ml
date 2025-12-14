# Project Architecture & Implementation Plan
**Topic:** Distributed Bitcoin Trend Prediction using Apache Spark
**Context:** Big Data Architecture Course Project

## 1. Project Scope
We are building a scalable ML pipeline to predict short-term Bitcoin price movement. The core objective is not just model accuracy, but demonstrating successful cluster management, distributed model training (Spark MLLib), and scalability analysis on the Google Cloud Platform (GCP).

---

## 2. Architecture & Data Flow

### Module A: Data Ingestion (ETL)
**Source:** OpenML (Bitcoin/Cryptocurrency Historical Data)  
**Sink:** Google Cloud Storage (GCS) Bucket

* **Fetch & Dump:**
    * Write a script to fetch the raw dataset from the OpenML API.
    * Dump the raw `.csv` or `.json` files directly into our designated GCS Bucket (Bronze Layer).
    * *Note: Ensure the bucket has the correct regional settings to minimize latency with our Dataproc cluster.*

### Module B: Distributed Preprocessing (Spark)
**Framework:** PySpark (Dataproc)

* **Schema Enforcement:**
    * Load data from GCS into Spark DataFrames.
    * Cast columns to strict types (`DoubleType` for prices, `TimestampType` for dates).
* **Handling Missing Values (Imputation):**
    * Since this is time-series data, we cannot simply drop rows without breaking the sequence.
    * **Strategy:** Apply **Backfill/Forward-fill** logic using Spark Window functions to propagate the last known valid price/volume to gaps.
* **Feature Engineering:**
    * Generate rolling window features (e.g., 7-day Moving Average, Volatility).
    * Target Variable Creation: `1` if $Price_{t+1} > Price_t$, else `0`.
    * **Vectorization:** Use `VectorAssembler` to squash features into a single vector column (required by MLLib).

---

## 3. Model Development
We will train three distinct models to compare performance and training overhead.

* **Model 1: Logistic Regression**
    * *Purpose:* Baseline linear model. Fast to train, easy to interpret.
* **Model 2: Random Forest Classifier**
    * *Purpose:* Parallel ensemble method. Good for capturing non-linearities without heavy tuning.
* **Model 3: Gradient Boosted Trees (GBT)**
    * *Purpose:* Sequential ensemble method. Likely higher accuracy but harder to parallelize; good for stress-testing the cluster.

---

## 4. Scalability Analysis & Cluster Management
This is the "Systems" part of the grade. We need to measure how the infrastructure handles increasing loads.

**Experimental Setup:**
We will run the full training pipeline four times, increasing the dataset size each time:

| Configuration | Dataset % | Objective |
| :--- | :--- | :--- |
| **Config A** | 25% | Smoke test; establish baseline latency. |
| **Config B** | 50% | Measure linearity of scaling. |
| **Config C** | 75% | Pre-production load test. |
| **Config D** | 100% | Full load; measure peak memory and CPU usage. |

* *Metric Capture:* For each run, we must log **Training Time**, **Inference Time**, and **Shuffle Read/Write** sizes from the Spark UI.

---

## 5. Visualization & Monitoring
* **Model Metrics:** Plot Accuracy/F1-Score across the three models.
* **System Metrics:** Visualize the "Scalability Curve" (Data Size vs. Training Time).
* **Monitoring:** Use the Spark History Server to take screenshots of the DAG (Directed Acyclic Graph) and stage timings for the final report.

---