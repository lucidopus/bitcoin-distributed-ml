import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

BUCKET_NAME = os.environ.get("BUCKET_NAME")
INPUT_FILE = f"gs://{BUCKET_NAME}/bitcoin_data_filled.csv"
OUTPUT_DIR = f"gs://{BUCKET_NAME}/feature_engineering_output"

def main():
    spark = SparkSession.builder \
        .appName("BitcoinFeatureEngineering") \
        .getOrCreate()
    
    print(f"Reading data from: {INPUT_FILE}")
    
    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME environment variable is not set")

    df = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)
    
    df = df.withColumn("Timestamp", F.to_timestamp(F.col("Timestamp")))
    df = df.orderBy("Timestamp")
    
    initial_count = df.count()
    print(f"Initial row count: {initial_count}")
    
    print("Calculating Advanced Features...")
    
    w_5 = Window.orderBy("Timestamp").rowsBetween(-5, -1)
    w_10 = Window.orderBy("Timestamp").rowsBetween(-10, -1)
    w_15 = Window.orderBy("Timestamp").rowsBetween(-15, -1)
    w_30 = Window.orderBy("Timestamp").rowsBetween(-30, -1)
    w_lag = Window.orderBy("Timestamp").rowsBetween(-1, -1)  
    
    
    df = df.withColumn("Prev_Close", F.lag("Close", 1).over(Window.orderBy("Timestamp")))
    df = df.withColumn("Log_Return", F.log(F.col("Close") / F.col("Prev_Close")))
    
    
    df = df.withColumn("SMA_5_Raw", F.avg("Close").over(w_5)) \
           .withColumn("SMA_10_Raw", F.avg("Close").over(w_10)) \
           .withColumn("SMA_15_Raw", F.avg("Close").over(w_15))
    
    
    df = df.withColumn("Feat_SMA_5", (F.col("Close") - F.col("SMA_5_Raw")) / F.col("SMA_5_Raw")) \
           .withColumn("Feat_SMA_10", (F.col("Close") - F.col("SMA_10_Raw")) / F.col("SMA_10_Raw")) \
           .withColumn("Feat_SMA_15", (F.col("Close") - F.col("SMA_15_Raw")) / F.col("SMA_15_Raw"))
    
    
    log_hl = F.log(F.col("High") / F.col("Low"))
    log_co = F.log(F.col("Close") / F.col("Open"))
    df = df.withColumn("Feat_GK_Vol", 0.5 * F.pow(log_hl, 2) - (0.38629 * F.pow(log_co, 2)))
    
    
    
    df = df.withColumn("Feat_Vol_Std", F.stddev("Log_Return").over(w_30))
    
    
    print("Calculating Target variable...")
    
    
    df = df.withColumn("FutureTimestamp", F.col("Timestamp") + F.expr("INTERVAL 15 MINUTES"))
    
    
    df_future = df.select(
        F.col("Timestamp").alias("FutureTimestamp"),
        F.col("Close").alias("FutureClose")
    )
    
    
    df = df.join(df_future, on="FutureTimestamp", how="left")
    
    
    df = df.withColumn("Target", 
        F.when(F.col("FutureClose") > F.col("Close"), 1).otherwise(0)
    )
    
    
    
    
    
    df_clean = df.dropna(subset=["Log_Return", "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
                                   "Feat_GK_Vol", "Feat_Vol_Std", "FutureClose", "Target"])
    
    final_count = df_clean.count()
    dropped_count = initial_count - final_count
    print(f"Dropped {dropped_count} rows due to insufficient history or no future data")
    print(f"Final row count: {final_count}")
    
    
    class_dist = df_clean.groupBy("Target").count().collect()
    print("Target distribution:")
    for row in class_dist:
        print(f"  Class {row['Target']}: {row['count']} ({100*row['count']/final_count:.1f}%)")
    
    
    df_final = df_clean.drop("SMA_5_Raw", "SMA_10_Raw", "SMA_15_Raw", 
                              "FutureClose", "FutureTimestamp", "Prev_Close", "Log_Return")
    
    
    print(f"Saving data to {OUTPUT_DIR}...")
    
    
    df_final.coalesce(1).write.mode("overwrite").option("header", "true").csv(OUTPUT_DIR)
    
    print("Job Complete.")
    spark.stop()

if __name__ == "__main__":
    main()
    