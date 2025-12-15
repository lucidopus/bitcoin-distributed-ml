import time
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    
    spark = SparkSession.builder.appName("Bitcoin_GBT_Training").getOrCreate()
    
    
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME env var not found")
        
    INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_feature_engineered.csv"
    RESULTS_PATH = f"gs://{BUCKET_NAME}/results/"
    
    print(f"--- STARTING GBT TRAINING (100% Data) ---")
    
    
    df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
    df = df.withColumn("Timestamp", F.to_timestamp("Timestamp"))
    
    
    feature_cols = [
        "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
        "Feat_GK_Vol", "Feat_Vol_Std", "Volume"
    ]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vec = assembler.transform(df)
    
    
    
    train_data, test_data = df_vec.randomSplit([0.8, 0.2], seed=42)
    
    train_count = train_data.count()
    print(f"Training on {train_count} rows...")

    
    print("Training Gradient Boosted Trees...")
    gbt = GBTClassifier(labelCol="Target", featuresCol="features", maxIter=10)
    
    start_time = time.time()
    model = gbt.fit(train_data)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Training Time: {duration:.2f} seconds")
    
    
    predictions = model.transform(test_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="Target", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print(f"Test Accuracy: {accuracy}")
    
    
    
    timestamp = int(time.time())
    result_record = {
        "model": "GBT",
        "rows": train_count,
        "time_seconds": duration,
        "accuracy": accuracy,
        "timestamp": timestamp
    }
    
    
    result_df = spark.createDataFrame([result_record])
    
    
    filename = f"gbt_run_{timestamp}"
    result_df.coalesce(1).write.mode("overwrite").json(f"{RESULTS_PATH}{filename}")
    
    print(f"Results saved to {RESULTS_PATH}{filename}")
    spark.stop()

if __name__ == "__main__":
    main()
    