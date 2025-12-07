# VS Code + GCP Integration - Complete Setup Guide

## 🎯 What You'll Get

✅ Write and edit code comfortably in VS Code  
✅ Submit jobs to GCP Dataproc from VS Code terminal  
✅ See real-time results in VS Code  
✅ Full scale-out testing (2, 3, 4 workers)  
✅ Test locally first, then run on GCP  

---

## 📋 PART 1: One-Time Setup (30 minutes)

### Step 1: Install Required Software

#### 1.1 Install VS Code
```
1. Go to: https://code.visualstudio.com/
2. Download for your OS
3. Install and open
```

#### 1.2 Install Python
```
Windows: https://www.python.org/downloads/
Mac: Already installed (or brew install python3)
Linux: sudo apt install python3 python3-pip
```

**Verify:**
```bash
python --version
# Should show Python 3.8+
```

#### 1.3 Install Google Cloud SDK

**Windows:**
```
1. Download: https://cloud.google.com/sdk/docs/install
2. Run GoogleCloudSDKInstaller.exe
3. Follow prompts (keep defaults)
4. Check "Run gcloud init" at end
```

**Mac:**
```bash
# Install Homebrew first if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install gcloud
brew install google-cloud-sdk
```

**Linux:**
```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
```

**Verify Installation:**
```bash
gcloud --version
# Should show version info
```

---

### Step 2: Install VS Code Extensions (2 minutes)

Open VS Code and install these extensions:

```
1. Click Extensions icon (left sidebar) or Ctrl+Shift+X
2. Search and install:
   - "Python" by Microsoft
   - "Cloud Code" by Google Cloud (optional but helpful)
   - "YAML" by Red Hat (for config files)
```

---

### Step 3: Authenticate with GCP (5 minutes)

#### 3.1 Open VS Code Terminal
```
View → Terminal (or Ctrl + `)
```

#### 3.2 Initialize gcloud
```bash
gcloud init
```

**You'll be prompted for:**
1. "Log in with your Google account?" → Yes
2. Browser opens → Select your Google account
3. "Allow Google Cloud SDK?" → Allow
4. Select or create a project

**OR if you already have an account:**

```bash
# Login
gcloud auth login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

#### 3.3 Set Default Region
```bash
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

**Verify Setup:**
```bash
gcloud config list
```

Should show your project, region, and account.

---

### Step 4: Create Project Structure in VS Code (3 minutes)

```bash
# In VS Code terminal
mkdir bitcoin-prediction
cd bitcoin-prediction
mkdir data scripts results notebooks
```

**Your structure:**
```
bitcoin-prediction/
├── data/
│   └── bitcoin_data.csv (will download)
├── scripts/
│   └── bitcoin_prediction.py (will create)
├── results/
│   └── (results will save here)
└── notebooks/
    └── (optional Jupyter notebooks)
```

---

### Step 5: Create Python Virtual Environment (Optional but Recommended)

```bash
# In VS Code terminal
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

**Install local testing packages (optional):**
```bash
pip install pyspark pandas numpy
```

---

## 📦 PART 2: GCP Setup (15 minutes)

### Step 1: Enable Required APIs

**In VS Code terminal:**

```bash
# Enable Dataproc
gcloud services enable dataproc.googleapis.com

# Enable Cloud Storage
gcloud services enable storage.googleapis.com

# Enable Compute Engine
gcloud services enable compute.googleapis.com

# Verify
gcloud services list --enabled | grep -E "dataproc|storage|compute"
```

---

### Step 2: Create Cloud Storage Bucket

```bash
# Get your project ID
PROJECT_ID=$(gcloud config get-value project)
echo "Project ID: $PROJECT_ID"

# Create bucket name
BUCKET_NAME="${PROJECT_ID}-bitcoin-data"
echo "Bucket name: $BUCKET_NAME"

# Create the bucket
gsutil mb -l us-central1 gs://${BUCKET_NAME}/

# Verify
gsutil ls
```

**✓ You should see:** `gs://your-project-bitcoin-data/`

---

### Step 3: Download Dataset

**Option A: Download to Local First**

```
1. Go to: https://www.openml.org/search?type=data&id=43947
2. Download bitcoin_data.csv
3. Save to: bitcoin-prediction/data/
```

**Option B: Download with Python**

Create `download_data.py` in VS Code:

```python
import openml
import pandas as pd

print("Downloading Bitcoin dataset from OpenML...")
dataset = openml.datasets.get_dataset(43947)
df = dataset.get_data()[0]
print(f"Downloaded {len(df)} rows")

df.to_csv('data/bitcoin_data.csv', index=False)
print("Saved to data/bitcoin_data.csv")
```

Run in VS Code terminal:
```bash
pip install openml
python download_data.py
```

---

### Step 4: Upload Dataset to GCP

```bash
# In VS Code terminal
gsutil cp data/bitcoin_data.csv gs://${BUCKET_NAME}/

# This takes 5-10 minutes for large file
# You'll see progress: Copying file://...

# Verify upload
gsutil ls gs://${BUCKET_NAME}/
```

**✓ Should show:** `gs://your-bucket/bitcoin_data.csv`

---

## 💻 PART 3: Create Your Code in VS Code (10 minutes)

### Step 1: Create Main Script

**In VS Code, create `scripts/bitcoin_prediction.py`:**

Click `File → New File` or create in terminal:
```bash
code scripts/bitcoin_prediction.py
```

**Copy this complete code:**

```python
#!/usr/bin/env python3
"""
Bitcoin Short-Term Trend Prediction - GCP Dataproc Version
Run from VS Code terminal with: gcloud dataproc jobs submit pyspark
"""

import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Configuration - get bucket from command line
BUCKET_NAME = sys.argv[1] if len(sys.argv) > 1 else "bitcoin-bucket"
DATA_PATH = f"gs://{BUCKET_NAME}/bitcoin_data.csv"
OUTPUT_PATH = f"gs://{BUCKET_NAME}/results/"

print(f"\n{'='*80}")
print(f"BITCOIN PREDICTION - RUNNING ON GCP DATAPROC")
print(f"{'='*80}")
print(f"Data: {DATA_PATH}")
print(f"Output: {OUTPUT_PATH}\n")

# Initialize Spark
spark = SparkSession.builder.appName("Bitcoin_Prediction").getOrCreate()

def preprocess_data(df):
    """Preprocess and clean data"""
    print("\n[1/5] Preprocessing data...")
    
    df = df.withColumn("datetime", from_unixtime(col("Timestamp")))
    
    # Forward fill for price columns
    window_spec = Window.orderBy("Timestamp")
    for col_name in ["Open", "Close", "High", "Low", "Weighted_Price"]:
        df = df.withColumn(
            col_name,
            when(col(col_name).isNull(), 
                 last(col(col_name), ignorenulls=True).over(window_spec))
            .otherwise(col(col_name))
        )
    
    # Fill volume with 0
    df = df.fillna(0, subset=["Volume_(BTC)", "Volume_(Currency)"])
    df = df.dropna()
    
    print(f"Cleaned data: {df.count()} rows")
    return df

def create_features(df):
    """Create technical indicators"""
    print("\n[2/5] Creating technical features...")
    
    # Define windows
    w5 = Window.orderBy("Timestamp").rowsBetween(-4, 0)
    w10 = Window.orderBy("Timestamp").rowsBetween(-9, 0)
    w30 = Window.orderBy("Timestamp").rowsBetween(-29, 0)
    
    # Price-based features
    df = df.withColumn("price_change_pct", 
                       (col("Close") - col("Open")) / col("Open") * 100)
    df = df.withColumn("high_low_spread", col("High") - col("Low"))
    df = df.withColumn("close_weighted_diff", col("Close") - col("Weighted_Price"))
    
    # Moving averages
    df = df.withColumn("MA_5", avg("Close").over(w5))
    df = df.withColumn("MA_10", avg("Close").over(w10))
    df = df.withColumn("MA_30", avg("Close").over(w30))
    
    # Momentum indicators
    df = df.withColumn("momentum_5", 
                       col("Close") - lag("Close", 5).over(Window.orderBy("Timestamp")))
    df = df.withColumn("momentum_10", 
                       col("Close") - lag("Close", 10).over(Window.orderBy("Timestamp")))
    
    # Volatility
    df = df.withColumn("volatility_5", stddev("Close").over(w5))
    df = df.withColumn("volatility_10", stddev("Close").over(w10))
    
    # Volume features
    df = df.withColumn("volume_MA_5", avg("Volume_(BTC)").over(w5))
    df = df.withColumn("volume_change", 
                       col("Volume_(BTC)") - lag("Volume_(BTC)", 1).over(Window.orderBy("Timestamp")))
    
    # RSI (Relative Strength Index)
    df = df.withColumn("price_diff", 
                       col("Close") - lag("Close", 1).over(Window.orderBy("Timestamp")))
    df = df.withColumn("gain", when(col("price_diff") > 0, col("price_diff")).otherwise(0))
    df = df.withColumn("loss", when(col("price_diff") < 0, -col("price_diff")).otherwise(0))
    df = df.withColumn("avg_gain", avg("gain").over(w10))
    df = df.withColumn("avg_loss", avg("loss").over(w10))
    df = df.withColumn("RSI", 
                       100 - (100 / (1 + col("avg_gain") / (col("avg_loss") + 0.0001))))
    
    # Target variable: UP (1) or DOWN (0)
    df = df.withColumn("next_close", 
                       lead("Close", 1).over(Window.orderBy("Timestamp")))
    df = df.withColumn("target", 
                       when(col("next_close") > col("Close"), 1).otherwise(0))
    
    df = df.dropna()
    
    print(f"Features created: {df.count()} rows")
    return df

def train_model(train_df, test_df, feature_cols, model_type="RandomForest"):
    """Train classification model with hyperparameter tuning"""
    print(f"\n[3/5] Training {model_type}...")
    
    # Feature assembly
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
    
    if model_type == "RandomForest":
        classifier = RandomForestClassifier(
            featuresCol="scaled_features", 
            labelCol="target", 
            seed=42
        )
        # Hyperparameter grid
        paramGrid = ParamGridBuilder() \
            .addGrid(classifier.numTrees, [50, 100]) \
            .addGrid(classifier.maxDepth, [5, 10]) \
            .build()
        pipeline = Pipeline(stages=[assembler, scaler, classifier])
        
    else:  # GBT
        classifier = GBTClassifier(
            featuresCol="features",
            labelCol="target",
            maxIter=50,
            seed=42
        )
        paramGrid = ParamGridBuilder() \
            .addGrid(classifier.maxDepth, [3, 5]) \
            .addGrid(classifier.stepSize, [0.1, 0.2]) \
            .build()
        pipeline = Pipeline(stages=[assembler, classifier])
    
    # Cross-validation
    evaluator = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        seed=42
    )
    
    # Train
    start = time.time()
    cv_model = cv.fit(train_df)
    train_time = time.time() - start
    
    # Predict
    predictions = cv_model.bestModel.transform(test_df)
    
    print(f"{model_type} trained in {train_time:.2f} seconds")
    return cv_model.bestModel, predictions, train_time

def evaluate_model(predictions, model_name):
    """Evaluate model with multiple metrics"""
    print(f"\n[4/5] Evaluating {model_name}...")
    
    # Binary classification metrics
    binary_eval = BinaryClassificationEvaluator(labelCol="target")
    auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})
    
    # Multiclass metrics
    multi_eval = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction")
    accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
    precision = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})
    recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})
    f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})
    
    # Print results
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Target: >0.65)")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f} (Target: >0.60)")
    
    return {
        "model": model_name,
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    """Main execution pipeline"""
    total_start = time.time()
    
    # Load data
    print("\nLoading Bitcoin data from GCS...")
    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)
    print(f"Loaded: {df.count()} rows")
    
    # Preprocess
    df = preprocess_data(df)
    
    # Create features
    df = create_features(df)
    
    # Define feature columns
    feature_cols = [
        "price_change_pct", "high_low_spread", "close_weighted_diff",
        "MA_5", "MA_10", "MA_30",
        "momentum_5", "momentum_10",
        "volatility_5", "volatility_10",
        "volume_MA_5", "volume_change", "RSI"
    ]
    
    # Train-test split
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"\nTrain set: {train_df.count()} rows")
    print(f"Test set: {test_df.count()} rows")
    
    # Train Random Forest
    rf_model, rf_pred, rf_time = train_model(train_df, test_df, feature_cols, "RandomForest")
    rf_metrics = evaluate_model(rf_pred, "Random Forest")
    rf_metrics["train_time"] = rf_time
    
    # Train GBT
    gbt_model, gbt_pred, gbt_time = train_model(train_df, test_df, feature_cols, "GBT")
    gbt_metrics = evaluate_model(gbt_pred, "GBT")
    gbt_metrics["train_time"] = gbt_time
    
    # Final results
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"\n{'Metric':<20} {'Random Forest':<20} {'GBT':<20}")
    print("-"*60)
    for metric in ["auc", "accuracy", "precision", "recall", "f1", "train_time"]:
        print(f"{metric:<20} {rf_metrics[metric]:<20.4f} {gbt_metrics[metric]:<20.4f}")
    
    print(f"\nTotal Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Check target achievement
    print(f"\n{'='*80}")
    print("TARGET ACHIEVEMENT")
    print(f"{'='*80}")
    
    best_precision = max(rf_metrics['precision'], gbt_metrics['precision'])
    best_f1 = max(rf_metrics['f1'], gbt_metrics['f1'])
    
    precision_met = "✓ MET" if best_precision > 0.65 else "✗ NOT MET"
    f1_met = "✓ MET" if best_f1 > 0.60 else "✗ NOT MET"
    
    print(f"Precision > 0.65: {precision_met} (Best: {best_precision:.4f})")
    print(f"F1-Score > 0.60: {f1_met} (Best: {best_f1:.4f})")
    
    # Save results to GCS
    results_df = spark.createDataFrame([rf_metrics, gbt_metrics])
    results_df.write.mode("overwrite").json(OUTPUT_PATH)
    
    print(f"\n[5/5] Complete! Results saved to: {OUTPUT_PATH}")
    
    spark.stop()

if __name__ == "__main__":
    main()
```

**Save the file** (Ctrl+S)

---

### Step 2: Upload Script to GCS

**In VS Code terminal:**

```bash
# Make sure you're in the right directory
cd bitcoin-prediction

# Upload script to Cloud Storage
gsutil cp scripts/bitcoin_prediction.py gs://${BUCKET_NAME}/scripts/

# Verify upload
gsutil ls gs://${BUCKET_NAME}/scripts/
```

**✓ Should see:** `gs://your-bucket/scripts/bitcoin_prediction.py`

---

## 🚀 PART 4: Run Jobs from VS Code (60 minutes total)

### Experiment 1: Create and Run 2-Worker Cluster

#### Create Cluster

```bash
# In VS Code terminal
gcloud dataproc clusters create bitcoin-cluster-2w \
    --region=us-central1 \
    --zone=us-central1-a \
    --master-machine-type=n1-standard-4 \
    --master-boot-disk-size=100 \
    --num-workers=2 \
    --worker-machine-type=n1-standard-4 \
    --worker-boot-disk-size=100 \
    --image-version=2.1-debian11
```

*Takes 2-3 minutes. You'll see output in VS Code!*

**✓ Wait for:** "Cluster created successfully"

---

#### Submit Job

```bash
# Submit PySpark job
gcloud dataproc jobs submit pyspark \
    gs://${BUCKET_NAME}/scripts/bitcoin_prediction.py \
    --cluster=bitcoin-cluster-2w \
    --region=us-central1 \
    -- ${BUCKET_NAME}
```

**You'll see real-time output in VS Code terminal!**

Progress updates like:
```
[1/5] Preprocessing data...
Cleaned data: 4830000 rows
[2/5] Creating technical features...
Features created: 4829970 rows
[3/5] Training RandomForest...
RandomForest trained in 856.23 seconds
[4/5] Evaluating Random Forest...
...
```

**Takes 15-20 minutes**

#### Save Results

**When complete, copy the results table from your terminal!**

Create `results/experiment_2workers.txt` in VS Code and paste:
```
Workers: 2
Total Time: 1245.67 seconds
Random Forest - Precision: 0.6512, F1: 0.6156
GBT - Precision: 0.6389, F1: 0.6089
```

---

### Experiment 2: 3-Worker Cluster

```bash
# Create cluster
gcloud dataproc clusters create bitcoin-cluster-3w \
    --region=us-central1 \
    --zone=us-central1-a \
    --master-machine-type=n1-standard-4 \
    --num-workers=3 \
    --worker-machine-type=n1-standard-4 \
    --worker-boot-disk-size=100 \
    --image-version=2.1-debian11

# Submit job
gcloud dataproc jobs submit pyspark \
    gs://${BUCKET_NAME}/scripts/bitcoin_prediction.py \
    --cluster=bitcoin-cluster-3w \
    --region=us-central1 \
    -- ${BUCKET_NAME}
```

*Copy results to `results/experiment_3workers.txt`*

---

### Experiment 3: 4-Worker Cluster

```bash
# Create cluster
gcloud dataproc clusters create bitcoin-cluster-4w \
    --region=us-central1 \
    --zone=us-central1-a \
    --master-machine-type=n1-standard-4 \
    --num-workers=4 \
    --worker-machine-type=n1-standard-4 \
    --worker-boot-disk-size=100 \
    --image-version=2.1-debian11

# Submit job
gcloud dataproc jobs submit pyspark \
    gs://${BUCKET_NAME}/scripts/bitcoin_prediction.py \
    --cluster=bitcoin-cluster-4w \
    --region=us-central1 \
    -- ${BUCKET_NAME}
```

*Copy results to `results/experiment_4workers.txt`*

---

## 🧹 PART 5: Cleanup (CRITICAL!)

### Delete All Clusters

**In VS Code terminal:**

```bash
# Delete all clusters (IMPORTANT - avoid charges!)
gcloud dataproc clusters delete bitcoin-cluster-2w --region=us-central1 --quiet
gcloud dataproc clusters delete bitcoin-cluster-3w --region=us-central1 --quiet
gcloud dataproc clusters delete bitcoin-cluster-4w --region=us-central1 --quiet

# Verify all deleted
gcloud dataproc clusters list --region=us-central1
```

**✓ Should show:** "Listed 0 items"

---

### Download Results from GCS

```bash
# Download results to local
gsutil cp -r gs://${BUCKET_NAME}/results/ results/from_gcs/

# View them in VS Code
code results/from_gcs/
```

---

## 📊 PART 6: Create Summary in VS Code

### Create Results Summary

**Create `results/summary.md` in VS Code:**

```markdown
# Bitcoin Prediction - Scale-Out Results

## Experiment Results

### 2 Workers
- Total Time: [YOUR TIME] seconds
- RF Precision: [YOUR VALUE]
- RF F1: [YOUR VALUE]
- GBT Precision: [YOUR VALUE]
- GBT F1: [YOUR VALUE]

### 3 Workers
- Total Time: [YOUR TIME] seconds
- RF Precision: [YOUR VALUE]
- RF F1: [YOUR VALUE]
- GBT Precision: [YOUR VALUE]
- GBT F1: [YOUR VALUE]

### 4 Workers
- Total Time: [YOUR TIME] seconds
- RF Precision: [YOUR VALUE]
- RF F1: [YOUR VALUE]
- GBT Precision: [YOUR VALUE]
- GBT F1: [YOUR VALUE]

## Analysis

### Speedup
- 3 workers vs 2: [CALCULATE]% faster
- 4 workers vs 2: [CALCULATE]% faster

### Target Achievement
- Precision > 65%: [YES/NO]
- F1 > 60%: [YES/NO]
```

---

## 🎯 Quick Command Reference

**Save these in VS Code for easy access!**

Create `commands.sh`:

```bash
#!/bin/bash

# Set variables
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET_NAME="${PROJECT_ID}-bitcoin-data"

# Upload script
alias upload-script="gsutil cp scripts/bitcoin_prediction.py gs://${BUCKET_NAME}/scripts/"

# Create cluster (replace X with worker count)
create-cluster() {
    gcloud dataproc clusters create bitcoin-cluster-$1w \
        --region=us-central1 \
        --zone=us-central1-a \
        --master-machine-type=n1-standard-4 \
        --num-workers=$1 \
        --worker-machine-type=n1-standard-4 \
        --worker-boot-disk-size=100 \
        --image-version=2.1-debian11
}

# Submit job
submit-job() {
    gcloud dataproc jobs submit pyspark \
        gs://${BUCKET_NAME}/scripts/bitcoin_prediction.py \
        --cluster=bitcoin-cluster-$1w \
        --region=us-central1 \
        -- ${BUCKET_NAME}
}

# Delete cluster
delete-cluster() {
    gcloud dataproc clusters delete bitcoin-cluster-$1w \
        --region=us-central1 --quiet
}

# List everything
alias list-clusters="gcloud dataproc clusters list --region=us-central1"
alias list-jobs="gcloud dataproc jobs list --region=us-central1 --limit=10"
alias list-bucket="gsutil ls gs://${BUCKET_NAME}/"
```

**Usage:**
```bash
source commands.sh
create-cluster 2
submit-job 2
delete-cluster 2
```

---

## 🎨 VS Code Tips

### Use Split View

```
1. Open script: scripts/bitcoin_prediction.py
2. Open terminal: View → Terminal
3. Drag terminal to right side
4. Now you see code + terminal output side-by-side!
```

### Monitor Job in Browser

```
1. While job runs in VS Code terminal
2. Open browser: https://console.cloud.google.com
3. Navigate: Dataproc → Jobs
4. Click your job → View Logs
5. See detailed progress
```

### Save Output to File

```bash
# Redirect output to file
gcloud dataproc jobs submit pyspark \
    gs://${BUCKET_NAME}/scripts/bitcoin_prediction.py \
    --cluster=bitcoin-cluster-2w \
    --region=us-central1 \
    -- ${BUCKET_NAME} 2>&1 | tee results/output_2workers.log
```

Now you have output in VS Code AND in a file!

---

## ✅ Final Checklist

- [ ] VS Code installed
- [ ] Google Cloud SDK installed
- [ ] Authenticated with `gcloud auth login`
- [ ] Project created and set
- [ ] Bucket created
- [ ] Dataset uploaded to GCS
- [ ] Script created in VS Code
- [ ] Script uploaded to GCS
- [ ] Can run `gcloud` commands from VS Code terminal
- [ ] Ready to create clusters and submit jobs!

---

## 🚀 You're Ready!

Your workflow:
1. ✅ Edit code in VS Code
2. ✅ Upload with `gsutil cp`
3. ✅ Create cluster with `gcloud dataproc clusters create`
4. ✅ Submit job with `gcloud dataproc jobs submit`
5. ✅ Watch output in VS Code terminal
6. ✅ Copy results to local files
7. ✅ Delete cluster with `gcloud dataproc clusters delete`

**Everything from VS Code - comfortable and powerful!** 🎯