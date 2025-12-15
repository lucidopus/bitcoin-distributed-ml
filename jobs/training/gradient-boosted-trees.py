import time
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder.appName("Bitcoin_GBT_Training").getOrCreate()

BUCKET_NAME = os.environ.get("BUCKET_NAME")
if not BUCKET_NAME:
    raise ValueError("BUCKET_NAME env var not found")

INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_scaled.csv"
RESULTS_PATH = f"gs://{BUCKET_NAME}/results/"

print(f"Reading data from {INPUT_PATH}...")
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
df = df.withColumn("Timestamp", F.to_timestamp("Timestamp"))
print(f"Data read successfully. Initial count: {df.count()}")
   
feature_cols = [
    "Open", "High", "Low", "Close",
    "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price",
    "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
    "Feat_GK_Vol", "Feat_Vol_Std"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_vec = assembler.transform(df)
print("Feature assembly complete.")

# Sequential train/test split (80/20)
window_spec = Window.orderBy("Timestamp")
df_with_row = df_vec.withColumn("row_idx", F.row_number().over(window_spec))
total_count = df_with_row.count()
train_count_limit = int(total_count * 0.8)

train_data = df_with_row.filter(F.col("row_idx") <= train_count_limit).drop("row_idx")
test_data = df_with_row.filter(F.col("row_idx") > train_count_limit).drop("row_idx")

train_count = train_data.count()
test_count = test_data.count()

print(f"Total rows: {total_count}")
print(f"Train row limit: {train_count_limit}")
print(f"Actual Training rows: {train_count}")
print(f"Actual Testing rows: {test_count}")

# Verify Temporal Split
train_max_date = train_data.agg(F.max("Timestamp")).collect()[0][0]
test_min_date = test_data.agg(F.min("Timestamp")).collect()[0][0]

print(f"Training Max Timestamp: {train_max_date}")
print(f"Testing Min Timestamp: {test_min_date}")

if train_max_date < test_min_date:
    print("SUCCESS: Training data strictly precedes testing data.")
else:
    print("WARNING: Data leakage detected! Training data overlaps or succeeds testing data.")

print(f"Data split complete. Training on {train_count} rows...")

gbt = GBTClassifier(labelCol="Target", featuresCol="features", maxIter=10)

start_time = time.time()
model = gbt.fit(train_data)
end_time = time.time()
duration = end_time - start_time

print(f"Training Time: {duration:.2f} seconds")
print("Model training complete.")

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="Target", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

print(f"Test Accuracy: {accuracy}")
print("Evaluation complete.")

# Generate Classification Report
predictionAndLabels = predictions.select("prediction", "Target").rdd.map(
    lambda row: (float(row.prediction), float(row.Target))
)
metrics = MulticlassMetrics(predictionAndLabels)

labels = [0.0, 1.0]
classification_report = {}

# Cache predictions to avoid recomputation
predictions.cache()

# Per-class metrics - FIX: Calculate support properly
class_0_support = predictions.filter(F.col("Target") == 0.0).count()
class_1_support = predictions.filter(F.col("Target") == 1.0).count()
total_support = predictions.count()

for i, label in enumerate(labels):
    support = class_0_support if i == 0 else class_1_support
    classification_report[str(int(label))] = {
        "precision": float(metrics.precision(label)),
        "recall": float(metrics.recall(label)),
        "f1-score": float(metrics.fMeasure(label, 1.0)),
        "support": support
    }

# Overall metrics - FIX: Convert to float and add support
classification_report["accuracy"] = float(accuracy)
classification_report["weighted avg"] = {
    "precision": float(metrics.weightedPrecision),
    "recall": float(metrics.weightedRecall),
    "f1-score": float(metrics.weightedFMeasure()),
    "support": total_support
}

timestamp = int(time.time())
result_record = {
    "model": "Gradient Boosted Trees",
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
