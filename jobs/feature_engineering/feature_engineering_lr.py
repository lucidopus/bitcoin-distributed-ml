import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import subprocess

# Environment Variables
def main():
    spark = SparkSession.builder.appName("Bitcoin_FeatureEngineering_LR").getOrCreate()

    # Resolve BUCKET_NAME
    BUCKET_NAME = os.environ.get("BUCKET_NAME")
    if not BUCKET_NAME:
        # Try retrieving from Spark Config (works for client mode if passed as spark.driverEnv)
        try:
            BUCKET_NAME = spark.conf.get("spark.driverEnv.BUCKET_NAME")
        except:
            pass
            
    if not BUCKET_NAME:
        # Try retrieving from AppMasterEnv (works for cluster mode)
        try:
            BUCKET_NAME = spark.conf.get("spark.yarn.appMasterEnv.BUCKET_NAME")
        except:
            pass

    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME env var is required (checked os.environ and spark.conf)")

    INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_feature_engineered.csv"
    OUTPUT_FILENAME = "bitcoin_data_for_lr.csv"
    OUTPUT_DIR_URI = f"gs://{BUCKET_NAME}/temp_lr_data"
    FINAL_OUTPUT_URI = f"gs://{BUCKET_NAME}/{OUTPUT_FILENAME}"
    
    print(f"Reading data from {INPUT_PATH}...")
    df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
    df = df.withColumn("Timestamp", F.to_timestamp("Timestamp"))
    
    # ---------------------------------------------------------
    # Generate Lag Features
    # ---------------------------------------------------------
    print("Generating Lag Features (1-5)...")
    
    w = Window.orderBy("Timestamp")
    
    lag_cols = []
    # Using 'Close' for lags. If feature_engineered is unscaled, 'Close' is raw price.
    for i in range(1, 6):
        col_name = f"Lag_{i}"
        df = df.withColumn(col_name, F.lag("Close", i).over(w))
        lag_cols.append(col_name)
        
    # ---------------------------------------------------------
    # SCALE DATA (StandardScaler)
    # ---------------------------------------------------------
    print("Scaling Features (StandardScaler)...")
    
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.functions import vector_to_array
    
    # Define all features to be scaled
    # We exclude Timestamp, Target, and any string columns
    # We want to scale: Open, High, Low, Close, Volume*, Weighted_Price, Feat_*, Lag_*
    
    # Get list of columns to scale
    # We know the schema from previous steps or we can infer.
    # Base columns in bitcoin_data_feature_engineered.csv usually are:
    # Open, High, Low, Close, Volume_(BTC), Volume_(Currency), Weighted_Price, 
    # Feat_SMA_5, Feat_SMA_10, Feat_SMA_15, Feat_GK_Vol, Feat_Vol_Std
    # Plus our new Lag_1..5
    
    feature_cols = [
        "Open", "High", "Low", "Close",
        "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price",
        "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
        "Feat_GK_Vol", "Feat_Vol_Std"
    ] + lag_cols
    
    print(f"Scaling {len(feature_cols)} features...")
    
    # CRITICAL FIX: VectorAssembler crashes on NULLs. 
    # Even though we dropped nulls for lags, other columns might have nulls.
    # We must strictly drop NA for ALL feature columns before assembling.
    final_count_pre_drop = df.count()
    df = df.dropna(subset=feature_cols)
    final_count_post_drop = df.count()
    print(f"Dropped {final_count_pre_drop - final_count_post_drop} rows with NULLs before VectorAssembler.")
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    assembled_df = assembler.transform(df)
    
    scaler = StandardScaler(inputCol="features_raw", outputCol="scaled_features", withMean=True, withStd=True)
    scaler_model = scaler.fit(assembled_df)
    scaled_df = scaler_model.transform(assembled_df)
    
    # Unpack vector to columns
    # We overwrite the original columns with their scaled versions
    
    scaled_df = scaled_df.withColumn("scaled_array", vector_to_array("scaled_features"))
    
    select_cols = ["Timestamp", "Target"] # Keep these as is
    
    # Add scaled columns
    for i, col_name in enumerate(feature_cols):
        # We rename the scaled value to the original name, effectively replacing it
        select_cols.append(F.col("scaled_array")[i].alias(col_name))
        
    df_final = scaled_df.select(*select_cols)
    
    print("Scaling complete. Schema of output:")
    df_final.printSchema()

    # ---------------------------------------------------------
    # Save Feature Engineered Dataset
    # ---------------------------------------------------------
    print(f"Saving prepared dataset to {FINAL_OUTPUT_URI}...")
    
    # Save as single CSV (Coalesce 1)
    df_final.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_DIR_URI)
    
    # Standard GCS move/rename Dance
    # (Similar to data-scaling.py logic to ensure nice filename)
    try:
        print("Renaming output file in GCS...")
        # List files in the output directory
        ls_cmd = f"gsutil ls {OUTPUT_DIR_URI}/*.csv"
        csv_files = subprocess.check_output(ls_cmd, shell=True).decode("utf-8").strip().split("\n")
        
        if csv_files and csv_files[0]:
            part_file = csv_files[0]
            print(f"Found part file: {part_file}")
            
            # Move to final location
            subprocess.check_call(f"gsutil mv {part_file} {FINAL_OUTPUT_URI}", shell=True)
            print(f"Successfully saved to {FINAL_OUTPUT_URI}")
            
            # Cleanup temp dir
            subprocess.check_call(f"gsutil -m rm -r {OUTPUT_DIR_URI}", shell=True)
            print("Temp directory cleaned up.")
        else:
            print("Error: No CSV file found in output directory.")
            
    except Exception as e:
        print(f"Error during GCS file operations: {e}")
        # Fallback: The data is in the directory, just maybe not renamed perfectly.
        
    spark.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
