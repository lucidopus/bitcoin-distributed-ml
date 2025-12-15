#!/usr/bin/env python3
"""
Standard Scaling for Bitcoin Minute-Level Data using PySpark on GCP Dataproc
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col
import subprocess
import sys

# GCP Configuration
CLUSTER_NAME = "bitcoin-cluster-w2"
REGION = "us-central1"
ZONE = "us-central1-a"
PROJECT_ID = "bitcoin-trend-prediction1"
BUCKET_NAME = "bitcoin-trend-prediction1-data"
INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_feature_engineered.csv"
OUTPUT_PATH = f"gs://{BUCKET_NAME}/data_scaling_output/" 

spark = SparkSession.builder \
    .appName("Bitcoin-Standard-Scaling") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

print("=" * 60)
print("Loading Bitcoin Data...")
print("=" * 60)

# Load data from GCS
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)

# Display schema and sample data
print("\nOriginal Schema:")
df.printSchema()
print(f"\nTotal records: {df.count()}")
print("\nSample data:")
df.show(5)

# Define feature columns (exclude Target and Timestamp)
feature_columns = [col_name for col_name in df.columns 
                    if col_name not in ['Timestamp', 'Target', 'target']]

print(f"\nFeature columns to scale: {feature_columns}")

# Handle missing values (if any)
print("\nChecking for missing values...")
df = df.na.drop()  # Drop rows with any null values

print(f"Records after cleaning: {df.count()}")

# Step 1: Assemble features into a vector
print("\n" + "=" * 60)
print("Step 1: Assembling features into vector...")
print("=" * 60)

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="skip"
)

assembled_df = assembler.transform(df)

# Step 2: Apply Standard Scaling
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

# Display scaling statistics
print("\nScaling Statistics:")
print(f"Mean: {scaler_model.mean}")
print(f"Std: {scaler_model.std}")

# Step 3: Convert vector back to individual columns
print("\n" + "=" * 60)
print("Step 3: Converting scaled vector to individual columns...")
print("=" * 60)

# Determine the target column name (case-insensitive)
target_col = 'Target' if 'Target' in df.columns else 'target'

# Convert the scaled_features vector back to individual columns
from pyspark.ml.functions import vector_to_array

# Convert vector to array first
scaled_with_array = scaled_df.withColumn(
    "scaled_array", 
    vector_to_array("scaled_features")
)

# Create individual columns for each scaled feature
output_df = scaled_with_array.select(
    'Timestamp',
    col(target_col).alias('Target'),
    *[col("scaled_array")[i].alias(feature_columns[i]) 
        for i in range(len(feature_columns))]
)

print("\nScaled data sample:")
output_df.show(5, truncate=False)

# Step 4: Save to GCS as single CSV file
print("\n" + "=" * 60)
print("Step 4: Saving scaled data to GCS as single CSV...")
print("=" * 60)

# Coalesce to single partition to create one CSV file
temp_csv_path = f"gs://{BUCKET_NAME}/temp_bitcoin_scaled"

# Spark writes to a directory. We write to a temporary directory first.
output_df.coalesce(1).write.mode("overwrite") \
    .option("header", "true") \
    .csv(temp_csv_path)

print(f"\n✓ Temporary data saved to: {temp_csv_path}")

# Step 5: Move and rename the output file
print("\n" + "=" * 60)
print("Step 5: Moving and renaming the part-file to its final destination...")
print("=" * 60)

try:
    # Find the part-file that Spark created in the temporary directory.
    # It's the only file that will have a .csv extension.
    find_cmd = f"gsutil ls {temp_csv_path}/*.csv"
    result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True, check=True)
    part_file = result.stdout.strip().split('\n')[0] # Get the first line in case of multiple matches
    
    if part_file:
        # Move the part-file to the final desired path and name.
        move_cmd = f"gsutil mv '{part_file}' {OUTPUT_PATH}"
        subprocess.run(move_cmd, shell=True, check=True)
        print(f"✓ Moved and renamed to {OUTPUT_PATH}")
        
        # Clean up the temporary directory Spark created.
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

# Display summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)
print(f"Input path: {INPUT_PATH}")
print(f"Output path: {OUTPUT_PATH}")
print(f"Total records processed: {output_df.count()}")
print(f"Features scaled: {len(feature_columns)}")
print(f"Target column: {target_col}")
print(f"Target distribution:")
output_df.groupBy('Target').count().show()

print("\n" + "=" * 60)
print("Processing Complete!")
print("=" * 60)

spark.stop()
