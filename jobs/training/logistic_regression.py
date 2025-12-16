import time
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.window import Window


def main():
    spark = SparkSession.builder.appName("Bitcoin_LogisticRegression_Training").getOrCreate()


    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    if not BUCKET_NAME:
        try:
            BUCKET_NAME = spark.conf.get("spark.driverEnv.BUCKET_NAME")
        except:
            pass
    if not BUCKET_NAME:
        try:
            BUCKET_NAME = spark.conf.get("spark.yarn.appMasterEnv.BUCKET_NAME")
        except:
            pass

    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME env var is required")

    INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_for_lr.csv"
    RESULTS_PATH = f"gs://{BUCKET_NAME}/results/"
    
    print(f"Reading data from {INPUT_PATH}...")
    try:
        df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
    except Exception as e:
        print(f"Error reading input file: {e}")
        print(f"Please ensure 'jobs/feature_engineering_lr.py' has been run first to generate {INPUT_PATH}")
        sys.exit(1)

    df = df.withColumn("Timestamp", F.to_timestamp("Timestamp"))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-percentage", type=float, help="Percentage of data to use (0-100)")
    args, _ = parser.parse_known_args()

    if args.data_percentage:
        percentage = args.data_percentage / 100.0
        print(f"DEBUG: Data percentage flag detected: {args.data_percentage}%")
        
        total_count = df.count()
        limit_count = int(total_count * percentage)
        
        print(f"DEBUG: Total records: {total_count}")
        print(f"DEBUG: Keeping top {args.data_percentage}% records.")
        print(f"DEBUG: Target record count: {limit_count}")
        

        w_filter = Window.orderBy("Timestamp")
        df = df.withColumn("row_num", F.row_number().over(w_filter))
        df = df.filter(F.col("row_num") <= limit_count).drop("row_num")
        
        print(f"DEBUG: Filtered records: {df.count()}")
    else:
        print("DEBUG: No data percentage limit applied. Using full dataset.")

    count = df.count()
    print(f"Total Rows: {count}")

    base_features = [
        "Open", "High", "Low", "Close",
        "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price",
        "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
        "Feat_GK_Vol", "Feat_Vol_Std"
    ]
    lag_features = ["Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5"]
    
    feature_cols = base_features + lag_features
    print(f"Using Features: {feature_cols}")
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    

    lr = LogisticRegression(
        featuresCol="features", 
        labelCol="Target", 
        maxIter=100
    )
    
    pipeline = Pipeline(stages=[assembler, lr])
    
    print("Performing Time-Based Train/Test Split...")
    w = Window.orderBy("Timestamp")
    df = df.withColumn("rank", F.percent_rank().over(w))
    
    train_data = df.filter(F.col("rank") <= 0.8).drop("rank")
    test_data = df.filter(F.col("rank") > 0.8).drop("rank")
    
    print(f"Training Rows: {train_data.count()}")
    print(f"Testing Rows:  {test_data.count()}")
    
    print("Training Model (Pipeline)...")
    start_time = time.time()
    
    model = pipeline.fit(train_data)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Training Time: {duration:.2f} seconds")
    
    predictions = model.transform(test_data)
    
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="Target", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="Target", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc = evaluator_auc.evaluate(predictions)
    print(f"Test AUC-ROC: {auc:.4f}")
    
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
    
    classification_report["accuracy"] = accuracy
    classification_report["weighted avg"] = {
        "precision": metrics.weightedPrecision,
        "recall": metrics.weightedRecall,
        "f1-score": metrics.weightedFMeasure(),
        "support": predictions.count()
    }
    
    timestamp = int(time.time())
    result_record = {
        "model": "Logistic Regression (AR)",
        "rows": train_data.count(),
        "time_seconds": duration,
        "accuracy": accuracy,
        "auc_roc": auc,
        "classification_report": classification_report,
        "features": feature_cols,
        "timestamp": timestamp
    }
    
    result_df = spark.createDataFrame([result_record])
    result_df.coalesce(1).write.mode("append").json(RESULTS_PATH)
    
    print(f"Results saved to {RESULTS_PATH}")
    spark.stop()

if __name__ == "__main__":
    main()
