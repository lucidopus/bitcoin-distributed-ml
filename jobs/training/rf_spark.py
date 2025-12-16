import time
import os
import json
import sys

IS_DATAPROC = os.path.exists('/usr/local/share/google/dataproc')

if not IS_DATAPROC:
    from google.cloud import dataproc_v1
    from google.cloud import storage

PROJECT_ID = "bitcoin-trend-prediction1"
REGION = "us-central1"
CLUSTER_NAME = "bitcoin-cluster-w3"
BUCKET_NAME = "bitcoin-trend-prediction1-data"

if IS_DATAPROC:
    print("Running on Dataproc cluster - starting training...")
    
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.mllib.evaluation import MulticlassMetrics
    
    # Initialize Spark
    spark = SparkSession.builder.appName("Bitcoin_RF_Training").getOrCreate()
    
    INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_scaled.csv"
    RESULTS_PATH = f"gs://{BUCKET_NAME}/results/"
    
    print(f"Reading data from {INPUT_PATH}...")
    df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
    df = df.withColumn("Timestamp", F.to_timestamp("Timestamp"))
    total_count = df.count()
    print(f"Total rows in dataset: {total_count}")

    df = df.orderBy(F.col("Timestamp").desc())
    df = df.limit(int(total_count * 1.0))

    sampled_count = df.count()
    print(f"Using 100% of data: {sampled_count} rows")
    
    # Feature columns
    feature_cols = [
        "Open", "High", "Low", "Close",
        "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price",
        "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
        "Feat_GK_Vol", "Feat_Vol_Std"
    ]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vec = assembler.transform(df)
    
    # Train/test split
    train_fraction = 0.8
    train_class_0 = df_vec.filter(F.col("Target") == 0).sample(fraction=train_fraction, seed=42)
    train_class_1 = df_vec.filter(F.col("Target") == 1).sample(fraction=train_fraction, seed=42)
    train_data = train_class_0.union(train_class_1)
    test_data = df_vec.join(train_data, on=df_vec.columns, how="left_anti")
    
    train_count = train_data.count()
    print(f"Training on {train_count} rows...")
    
    rf = RandomForestClassifier(
        labelCol="Target",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42
    )
    
    start_time = time.time()
    model = rf.fit(train_data)
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Training Time: {duration:.2f} seconds")
    
    predictions = model.transform(test_data)
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol="Target", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    predictionAndLabels = predictions.select("prediction", "Target").rdd.map(
        lambda row: (float(row.prediction), float(row.Target))
    )
    metrics = MulticlassMetrics(predictionAndLabels)
    
    labels = [0.0, 1.0]
    classification_report = {}
    
    for label in labels:
        classification_report[str(int(label))] = {
            "precision": metrics.precision(label),
            "recall": metrics.recall(label),
            "f1-score": metrics.fMeasure(label, 1.0),
            "support": predictions.filter(F.col("Target") == label).count()
        }
    
    classification_report["accuracy"] = metrics.accuracy
    classification_report["weighted avg"] = {
        "precision": metrics.weightedPrecision,
        "recall": metrics.weightedRecall,
        "f1-score": metrics.weightedFMeasure(),
        "support": predictions.count()
    }
    
    # Save results
    timestamp = int(time.time())
    result_record = {
        "model": "Random Forest",
        "rows": train_count,
        "time_seconds": duration,
        "accuracy": accuracy,
        "classification_report": classification_report,
        "timestamp": timestamp
    }
    
    result_df = spark.createDataFrame([result_record])
    result_df.coalesce(1).write.mode("append").json(RESULTS_PATH)
    
    print(f"Results saved to {RESULTS_PATH}")
    spark.stop()

else:
    print(f"Submitting job to existing cluster: {CLUSTER_NAME}")
    print(f"Project: {PROJECT_ID}")
    print(f"Region: {REGION}")
    print(f"Bucket: {BUCKET_NAME}\n")
    
    script_path = os.path.abspath(__file__)
    gcs_script_path = "scripts/bitcoin_rf_training.py"
    
    print(f"[1/2] Uploading script to GCS...")
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_script_path)
    blob.upload_from_filename(script_path)
    print(f"Script uploaded to gs://{BUCKET_NAME}/{gcs_script_path}")
    
    print(f"\n[2/2] Submitting job to cluster...")
    try:
        job_client = dataproc_v1.JobControllerClient(
            client_options={"api_endpoint": f"{REGION}-dataproc.googleapis.com:443"}
        )
        
        job_config = {
            "placement": {
                "cluster_name": CLUSTER_NAME
            },
            "pyspark_job": {
                "main_python_file_uri": f"gs://{BUCKET_NAME}/{gcs_script_path}",
                "properties": {
                    "spark.executor.memory": "3g",
                    "spark.executor.cores": "2",
                    "spark.driver.memory": "3g"
                }
            }
        }
        
        print(f"Submitting PySpark job to cluster {CLUSTER_NAME}...")
        operation = job_client.submit_job_as_operation(
            request={
                "project_id": PROJECT_ID,
                "region": REGION,
                "job": job_config
            }
        )
        
        print("Job submitted. Waiting for completion...")
        result = operation.result()
        print(f"Job finished with state: {result.status.state.name}")
        
        print("JOB COMPLETED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\nJob submission failed: {e}")
        raise