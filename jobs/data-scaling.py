#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col
import subprocess
import sys
import os

CLUSTER_NAME = os.environ.get("CLUSTER_NAME")
REGION = os.environ.get("REGION")
ZONE = os.environ.get("ZONE")
PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_feature_engineered.csv"
OUTPUT_PATH = f"gs://{BUCKET_NAME}/data_scaling_output/" 

spark = SparkSession.builder \
    .appName("Bitcoin-Standard-Scaling") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

print("Loading Bitcoin Data...")

df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)

# Display schema
print("\nOriginal Schema:")
df.printSchema()
print(f"\nTotal records: {df.count()}")

# Define feature columns
feature_columns = [col_name for col_name in df.columns 
                    if col_name not in ['Timestamp', 'Target', 'target']]

print(f"\nFeature columns to scale: {feature_columns}")

# Handle missing values (if any)
print("\nChecking for missing values...")
df = df.na.drop()

print(f"Records after cleaning: {df.count()}")

print("Step 1: Assembling features into vector...")

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="skip"
)

assembled_df = assembler.transform(df)

print("\n" + "=" * 60)
print("Step 2: Applying Standard Scaling...")
print("=" * 60)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withMean=True,
    withStd=True
)

# Fit the scaler on the data
scaler_model = scaler.fit(assembled_df)

# Transform the data
scaled_df = scaler_model.transform(assembled_df)

print("\nScaling Statistics:")
print(f"Mean: {scaler_model.mean}")
print(f"Std: {scaler_model.std}")

print("Step 3: Converting scaled vector to individual columns...")
target_col = 'Target' if 'Target' in df.columns else 'target'
from pyspark.ml.functions import vector_to_array
scaled_with_array = scaled_df.withColumn(
    "scaled_array", 
    vector_to_array("scaled_features")
)

output_df = scaled_with_array.select(
    'Timestamp',
    col(target_col).alias('Target'),
    *[col("scaled_array")[i].alias(feature_columns[i]) 
        for i in range(len(feature_columns))]
)

print("\nScaled data sample:")
output_df.show(5, truncate=False)

print("Step 4: Saving scaled data to GCS as single CSV...")

temp_csv_path = f"gs://{BUCKET_NAME}/temp_bitcoin_scaled"

output_df.coalesce(1).write.mode("overwrite") \
    .option("header", "true") \
    .csv(temp_csv_path)

print(f"\n✓ Temporary data saved to: {temp_csv_path}")

print("\n" + "=" * 60)
print("Step 5: Moving and renaming the part-file to its final destination...")
print("=" * 60)

try:
    find_cmd = f"gsutil ls {temp_csv_path}/*.csv"
    result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True, check=True)
    part_file = result.stdout.strip().split('\n')[0] 
    
    if part_file:
        move_cmd = f"gsutil mv '{part_file}' {OUTPUT_PATH}"
        subprocess.run(move_cmd, shell=True, check=True)
        print(f"✓ Moved and renamed to {OUTPUT_PATH}")
        
        delete_cmd = f"gsutil -m rm -r {temp_csv_path}"
        subprocess.run(delete_cmd, shell=True, check=True)
        print(f"✓ Deleted temporary folder: {temp_csv_path}")
        print(f"\n✓ Final CSV file saved at: {OUTPUT_PATH}")
    else:
        print("✗ Error: Could not find the Spark output file to move.")

except subprocess.CalledProcessError as e:
    print(f"✗ An error occurred during file operations with gsutil: {e}")
    print(f"  - Command: {e.cmd}")
    print(f"  - stdout: {e.stdout}")
    print(f"  - stderr: {e.stderr}")

print("Summary Statistics")
print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Total records processed: {output_df.count()}")
print(f"Features scaled: {len(feature_columns)}")
print(f"Target column: {target_col}")
print(f"Target distribution:")
output_df.groupBy('Target').count().show()
print("Processing Complete!")

spark.stop()
