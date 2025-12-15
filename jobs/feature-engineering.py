# import os
# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F
# from pyspark.sql.window import Window

# # --- CONFIGURATION ---
# BUCKET_NAME = os.environ.get("BUCKET_NAME")
# INPUT_FILE = f"gs://{BUCKET_NAME}/bitcoin_data_filled.csv"
# OUTPUT_DIR = f"gs://{BUCKET_NAME}/feature_engineering_output"

# def main():
#     spark = SparkSession.builder \
#         .appName("BitcoinFeatureEngineering") \
#         .getOrCreate()

#     print(f"Reading data from: {INPUT_FILE}")

#     # 1. Load Data
#     df = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)

#     # 2. Cleanup & Time Handling
#     # Ensure Timestamp is valid
#     df = df.withColumn("Timestamp", F.to_timestamp(F.col("Timestamp")))
#     df = df.orderBy("Timestamp")

#     # 3. Feature Engineering
#     print("Calculating Advanced Features...")
    
#     # Define Windows
#     w_5 = Window.orderBy("Timestamp").rowsBetween(-4, 0)
#     w_10 = Window.orderBy("Timestamp").rowsBetween(-9, 0)
#     w_15 = Window.orderBy("Timestamp").rowsBetween(-14, 0)
#     w_30 = Window.orderBy("Timestamp").rowsBetween(-29, 0)
#     w_target = Window.orderBy("Timestamp").rowsBetween(15, 15) # Look ahead 15 mins

#     # Calculate Raw SMAs (Intermediate)
#     df = df.withColumn("SMA_5_Raw", F.avg("Close").over(w_5)) \
#            .withColumn("SMA_10_Raw", F.avg("Close").over(w_10)) \
#            .withColumn("SMA_15_Raw", F.avg("Close").over(w_15))

#     # Calculate Relative Features (The ones we want)
#     df = df.withColumn("Feat_SMA_5", (F.col("Close") - F.col("SMA_5_Raw")) / F.col("SMA_5_Raw")) \
#            .withColumn("Feat_SMA_10", (F.col("Close") - F.col("SMA_10_Raw")) / F.col("SMA_10_Raw")) \
#            .withColumn("Feat_SMA_15", (F.col("Close") - F.col("SMA_15_Raw")) / F.col("SMA_15_Raw"))

#     # Volatility (Garman-Klass)
#     log_hl = F.log(F.col("High") / F.col("Low"))
#     log_co = F.log(F.col("Close") / F.col("Open"))
#     df = df.withColumn("Feat_GK_Vol", 0.5 * F.pow(log_hl, 2) - (0.38629 * F.pow(log_co, 2)))
#     df = df.withColumn("Feat_Vol_Std", F.stddev("Close").over(w_30))
    
#     # Fill NaNs
#     df = df.fillna(0, subset=["Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", "Feat_GK_Vol", "Feat_Vol_Std"])

#     # Target
#     df = df.withColumn("FutureClose", F.max("Close").over(w_target))
#     df = df.withColumn("Target", F.when(F.col("FutureClose") > F.col("Close"), 1).otherwise(0))
    
#     # Drop rows without targets (the last 15 mins of data)
#     df_clean = df.dropna(subset=["FutureClose"])

#     # 4. Cleanup
#     # Drop ONLY the intermediate calculation columns
#     # We KEEP "Open", "High", "Low", "Close", "Volume" + All New Features
#     df_final = df_clean.drop("SMA_5_Raw", "SMA_10_Raw", "SMA_15_Raw", "FutureClose")

#     # 5. Save Output
#     print(f"Saving temp data to {OUTPUT_DIR}...")
    
#     # coalesce(1) forces it to write a SINGLE part file inside the folder
#     df_final.coalesce(1) \
#         .write \
#         .mode("overwrite") \
#         .option("header", "true") \
#         .csv(OUTPUT_DIR)
    
#     print("Job Complete.")
#     spark.stop()

# if __name__ == "__main__":
#     main()

import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# --- CONFIGURATION ---
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INPUT_FILE = f"gs://{BUCKET_NAME}/bitcoin_data_filled.csv"
OUTPUT_FILE = f"gs://{BUCKET_NAME}/bitcoin_data_feature_engineered.csv"

def main():
    spark = SparkSession.builder \
        .appName("BitcoinFeatureEngineering") \
        .getOrCreate()
    
    print(f"Reading data from: {INPUT_FILE}")
    
    # 1. Load Data
    df = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)
    
    # 2. Cleanup & Time Handling
    df = df.withColumn("Timestamp", F.to_timestamp(F.col("Timestamp")))
    df = df.orderBy("Timestamp")
    
    initial_count = df.count()
    print(f"Initial row count: {initial_count}")
    
    # 3. Feature Engineering
    print("Calculating Advanced Features...")
    
    # FIXED: Windows only look at PAST data (exclude current row)
    w_5 = Window.orderBy("Timestamp").rowsBetween(-5, -1)
    w_10 = Window.orderBy("Timestamp").rowsBetween(-10, -1)
    w_15 = Window.orderBy("Timestamp").rowsBetween(-15, -1)
    w_30 = Window.orderBy("Timestamp").rowsBetween(-30, -1)
    w_lag = Window.orderBy("Timestamp").rowsBetween(-1, -1)  # Previous row
    
    # CRITICAL FIX: Calculate log returns (percentage changes) instead of raw prices
    # This makes volatility comparable across all price levels
    df = df.withColumn("Prev_Close", F.lag("Close", 1).over(Window.orderBy("Timestamp")))
    df = df.withColumn("Log_Return", F.log(F.col("Close") / F.col("Prev_Close")))
    
    # Calculate Raw SMAs on Close prices (for momentum indicators)
    df = df.withColumn("SMA_5_Raw", F.avg("Close").over(w_5)) \
           .withColumn("SMA_10_Raw", F.avg("Close").over(w_10)) \
           .withColumn("SMA_15_Raw", F.avg("Close").over(w_15))
    
    # Calculate Relative Features (The ones we want)
    df = df.withColumn("Feat_SMA_5", (F.col("Close") - F.col("SMA_5_Raw")) / F.col("SMA_5_Raw")) \
           .withColumn("Feat_SMA_10", (F.col("Close") - F.col("SMA_10_Raw")) / F.col("SMA_10_Raw")) \
           .withColumn("Feat_SMA_15", (F.col("Close") - F.col("SMA_15_Raw")) / F.col("SMA_15_Raw"))
    
    # Volatility (Garman-Klass) - Already uses log ratios, so it's scale-invariant ✓
    log_hl = F.log(F.col("High") / F.col("Low"))
    log_co = F.log(F.col("Close") / F.col("Open"))
    df = df.withColumn("Feat_GK_Vol", 0.5 * F.pow(log_hl, 2) - (0.38629 * F.pow(log_co, 2)))
    
    # CRITICAL FIX: Rolling standard deviation of LOG RETURNS, not raw prices
    # This measures volatility as percentage moves, comparable across all time periods
    df = df.withColumn("Feat_Vol_Std", F.stddev("Log_Return").over(w_30))
    
    # FIXED: Target calculation using time-based approach
    print("Calculating Target variable...")
    
    # Add a column for 15 minutes in the future
    df = df.withColumn("FutureTimestamp", F.col("Timestamp") + F.expr("INTERVAL 15 MINUTES"))
    
    # Create a future dataframe to join
    df_future = df.select(
        F.col("Timestamp").alias("FutureTimestamp"),
        F.col("Close").alias("FutureClose")
    )
    
    # Join to get the close price 15 minutes later
    df = df.join(df_future, on="FutureTimestamp", how="left")
    
    # Create binary target: 1 if price goes up, 0 if down
    df = df.withColumn("Target", 
        F.when(F.col("FutureClose") > F.col("Close"), 1).otherwise(0)
    )
    
    # FIXED: Drop rows where we can't calculate features or target
    # - First row (can't calculate log return)
    # - First 30 rows (can't calculate w_30 features)
    # - Last rows where FutureClose is null
    df_clean = df.dropna(subset=["Log_Return", "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
                                   "Feat_GK_Vol", "Feat_Vol_Std", "FutureClose", "Target"])
    
    final_count = df_clean.count()
    dropped_count = initial_count - final_count
    print(f"Dropped {dropped_count} rows due to insufficient history or no future data")
    print(f"Final row count: {final_count}")
    
    # Check class balance
    class_dist = df_clean.groupBy("Target").count().collect()
    print("Target distribution:")
    for row in class_dist:
        print(f"  Class {row['Target']}: {row['count']} ({100*row['count']/final_count:.1f}%)")
    
    # 4. Cleanup - Drop intermediate columns
    df_final = df_clean.drop("SMA_5_Raw", "SMA_10_Raw", "SMA_15_Raw", 
                              "FutureClose", "FutureTimestamp", "Prev_Close", "Log_Return")
    
    # 5. Save Output
    print(f"Saving data to {OUTPUT_FILE}...")
    
    # Write as a single CSV file
    temp_dir = f"gs://{BUCKET_NAME}/temp_feature_engineering/"
    df_final.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_dir)
    
    # Find the part file and move it
    from pyspark import SparkFiles
    import subprocess
    
    # Use Hadoop filesystem to rename
    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_config)
    
    # Find the part file
    src_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(temp_dir)
    file_status = fs.listStatus(src_path)
    
    for status in file_status:
        path = status.getPath().toString()
        if "part-" in path and path.endswith(".csv"):
            dst_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(OUTPUT_FILE)
            fs.rename(spark.sparkContext._jvm.org.apache.hadoop.fs.Path(path), dst_path)
            print(f"Renamed to: {OUTPUT_FILE}")
            break
    
    # Clean up temp directory
    fs.delete(src_path, True)
    
    print("Job Complete.")
    spark.stop()

if __name__ == "__main__":
    main()
    