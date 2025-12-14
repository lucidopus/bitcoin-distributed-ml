import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# --- CONFIGURATION ---
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INPUT_FILE = f"gs://{BUCKET_NAME}/feature_engineering_output.csv"
MODEL_OUTPUT = f"gs://{BUCKET_NAME}/models/random_forest_model"

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("BitcoinRandomForestModel") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

print("BTC PRICE PREDICTION - RANDOM FOREST MODEL")
print("="*60)
print(f"Reading data from: {INPUT_FILE}")

# 1. Load Feature-Engineered Data
df = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)

print(f"Initial data loaded: {df.count()} rows")
df.printSchema()

# 2. Define Feature Columns
feature_cols = [
    'Open', 'High', 'Low', 'Close',
    'Volume_BTC', 'Volume_Currency', 'Weighted_Price',
    'Feat_SMA_5', 'Feat_SMA_10', 'Feat_SMA_15',
    'Feat_GK_Vol', 'Feat_Vol_Std'
]

# 3. Data Cleaning
print("\nCleaning data...")
df = df.na.drop()

print(f"Data after cleaning: {df.count()} rows")

# Show target distribution
print("\nTarget Distribution:")
df.groupBy("Target").count().show()

# 4. Build ML Pipeline
print("\nBuilding ML Pipeline...")

# Assemble features into a vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="skip"
)

# Scale features
scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withStd=True,
    withMean=True
)

# Random Forest Classifier
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="Target",
    predictionCol="prediction",
    probabilityCol="probability",
    numTrees=100,
    maxDepth=10,
    maxBins=32,
    minInstancesPerNode=1,
    seed=42,
    subsamplingRate=0.8,
    featureSubsetStrategy="auto"
)

# Create pipeline
pipeline = Pipeline(stages=[assembler, scaler, rf])

# 5. Split Data (80-20)
print("\nSplitting data into train/test sets...")
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

print(f"Training set size: {train_data.count()}")
print(f"Test set size: {test_data.count()}")

# 6. Train Model
print("\n" + "="*60)
print("TRAINING RANDOM FOREST MODEL...")
print("="*60)
model = pipeline.fit(train_data)
print("✓ Model training completed!")

# 7. Make Predictions
print("\nMaking predictions on test data...")
predictions = model.transform(test_data)
predictions.cache()

# Show sample predictions
print("\nSample Predictions:")
predictions.select("Target", "prediction", "probability").show(20, truncate=False)

# 8. Evaluate Model
print("\n" + "="*60)
print("MODEL EVALUATION METRICS")
print("="*60)

# Binary Classification Metrics (AUC-ROC)
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="Target",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = binary_evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# Multiclass Metrics
mc_evaluator = MulticlassClassificationEvaluator(
    labelCol="Target",
    predictionCol="prediction"
)

accuracy = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "accuracy"})
precision = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedPrecision"})
recall = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "weightedRecall"})
f1 = mc_evaluator.evaluate(predictions, {mc_evaluator.metricName: "f1"})

print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted Precision: {precision:.4f}")
print(f"Weighted Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
predictions.groupBy("Target", "prediction").count().orderBy("Target", "prediction").show()

# 9. Feature Importance
rf_model = model.stages[-1]
feature_importance = rf_model.featureImportances

print("\n" + "="*60)
print("FEATURE IMPORTANCES")
print("="*60)

feature_importance_list = []
for idx, importance in enumerate(feature_importance):
    if idx < len(feature_cols):
        feature_importance_list.append((feature_cols[idx], float(importance)))

# Sort by importance
feature_importance_list.sort(key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance_list:
    print(f"{feature:20s}: {importance:.4f}")

# 10. Save Model
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)
print(f"Saving model to: {MODEL_OUTPUT}")
model.write().overwrite().save(MODEL_OUTPUT)
print("✓ Model saved successfully!")

# 11. Save Predictions Sample
predictions_output = f"gs://{BUCKET_NAME}/predictions/predictions_sample"
print(f"\nSaving sample predictions to: {predictions_output}")
predictions.select("Timestamp", "Close", "Target", "prediction", "probability") \
    .limit(1000) \
    .coalesce(1) \
    .write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(predictions_output)

predictions.unpersist()

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)

spark.stop()