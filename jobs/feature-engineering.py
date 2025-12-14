from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# --- CONFIGURATION ---
BUCKET_NAME = "bitcoin-trend-prediction1-data"
INPUT_FILE = f"gs://{BUCKET_NAME}/bitcoin_data_filled.csv"
OUTPUT_DIR = f"gs://{BUCKET_NAME}/feature_engineering_output"

def main():
    spark = SparkSession.builder \
        .appName("BitcoinFeatureEngineering") \
        .getOrCreate()

    print(f"Reading data from: {INPUT_FILE}")

    # 1. Load Data
    df = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)

    # 2. Cleanup & Time Handling
    # Ensure Timestamp is valid
    df = df.withColumn("Timestamp", F.to_timestamp(F.col("Timestamp")))
    df = df.orderBy("Timestamp")

    # 3. Feature Engineering
    print("Calculating Advanced Features...")
    
    # Define Windows
    w_5 = Window.orderBy("Timestamp").rowsBetween(-4, 0)
    w_10 = Window.orderBy("Timestamp").rowsBetween(-9, 0)
    w_15 = Window.orderBy("Timestamp").rowsBetween(-14, 0)
    w_30 = Window.orderBy("Timestamp").rowsBetween(-29, 0)
    w_target = Window.orderBy("Timestamp").rowsBetween(15, 15) # Look ahead 15 mins

    # Calculate Raw SMAs (Intermediate)
    df = df.withColumn("SMA_5_Raw", F.avg("Close").over(w_5)) \
           .withColumn("SMA_10_Raw", F.avg("Close").over(w_10)) \
           .withColumn("SMA_15_Raw", F.avg("Close").over(w_15))

    # Calculate Relative Features (The ones we want)
    df = df.withColumn("Feat_SMA_5", (F.col("Close") - F.col("SMA_5_Raw")) / F.col("SMA_5_Raw")) \
           .withColumn("Feat_SMA_10", (F.col("Close") - F.col("SMA_10_Raw")) / F.col("SMA_10_Raw")) \
           .withColumn("Feat_SMA_15", (F.col("Close") - F.col("SMA_15_Raw")) / F.col("SMA_15_Raw"))

    # Volatility (Garman-Klass)
    log_hl = F.log(F.col("High") / F.col("Low"))
    log_co = F.log(F.col("Close") / F.col("Open"))
    df = df.withColumn("Feat_GK_Vol", 0.5 * F.pow(log_hl, 2) - (0.38629 * F.pow(log_co, 2)))
    df = df.withColumn("Feat_Vol_Std", F.stddev("Close").over(w_30))
    
    # Fill NaNs
    df = df.fillna(0, subset=["Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", "Feat_GK_Vol", "Feat_Vol_Std"])

    # Target
    df = df.withColumn("FutureClose", F.max("Close").over(w_target))
    df = df.withColumn("Target", F.when(F.col("FutureClose") > F.col("Close"), 1).otherwise(0))
    
    # Drop rows without targets (the last 15 mins of data)
    df_clean = df.dropna(subset=["FutureClose"])

    # 4. Cleanup
    # Drop ONLY the intermediate calculation columns
    # We KEEP "Open", "High", "Low", "Close", "Volume" + All New Features
    df_final = df_clean.drop("SMA_5_Raw", "SMA_10_Raw", "SMA_15_Raw", "FutureClose")

    # 5. Save Output
    print(f"Saving temp data to {OUTPUT_DIR}...")
    
    # coalesce(1) forces it to write a SINGLE part file inside the folder
    df_final.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(OUTPUT_DIR)
    
    print("Job Complete.")
    spark.stop()

if __name__ == "__main__":
    main()