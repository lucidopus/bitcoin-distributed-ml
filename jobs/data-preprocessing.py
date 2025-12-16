from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import *
import subprocess
import os

PROJECT_ID = os.environ.get("PROJECT_ID")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INPUT_FILE = "bitcoin_data.csv"
OUTPUT_FILE = "bitcoin_data_filled.csv"
KAGGLE_FILE = "btcusd_1-min_data.csv"

def create_spark_session():
    """
    Create Spark session - automatically detects Dataproc environment
    """
    spark = SparkSession.builder \
        .appName("Bitcoin Data Fill - Dataproc") \
        .getOrCreate()
    
    is_dataproc = os.path.exists('/usr/local/share/google/dataproc')
    
    if is_dataproc:
        print("\n" + "="*70)
        print("✓ Running on Dataproc Cluster")
        print("✓ Spark UI available via Dataproc Console")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("⚠ Running in LOCAL mode")
        print("="*70 + "\n")
    
    return spark

def load_kaggle_data(spark, bucket_name, kaggle_file):
    """Load and prepare Kaggle Bitcoin data from GCS"""
    print(f"\n{'='*70}")
    print("JOB 1-5: Loading Kaggle data from GCS")
    print(f"{'='*70}")
    
    gcs_path = f"gs://{bucket_name}/{kaggle_file}"
    print(f"Reading from: {gcs_path}")
    
    df_kaggle = spark.read.csv(gcs_path, header=True, inferSchema=True)
    
    print("Converting Unix timestamp to datetime...")
    df_kaggle = df_kaggle.withColumn(
        "Timestamp",
        F.from_unixtime(F.col("Timestamp")).cast(TimestampType())
    )
    
    print("Renaming and creating columns...")
    df_kaggle = df_kaggle.withColumnRenamed("Volume", "Volume_(BTC)")
    
    df_kaggle = df_kaggle.withColumn(
        "Volume_(Currency)",
        F.col("Volume_(BTC)") * F.col("Close")
    )
    
    df_kaggle = df_kaggle.withColumn(
        "Weighted_Price",
        F.col("Close")
    )
    
    print("Counting Kaggle rows...")
    total_before = df_kaggle.count()
    print(f"✓ Total Kaggle rows: {total_before:,}")
    
    print("Filtering rows with volume > 0...")
    df_kaggle = df_kaggle.filter(F.col("Volume_(BTC)") > 0)
    
    total_after = df_kaggle.count()
    print(f"✓ After filtering: {total_after:,} rows")
    
    print("Sample data:")
    df_kaggle.show(5, truncate=False)
    
    print(f"{'='*70}\n")
    return df_kaggle

def load_original_data(spark, bucket_name, input_file):
    """Load original Bitcoin data from GCS"""
    print(f"\n{'='*70}")
    print("JOB 6-11: Loading original data from GCS")
    print(f"{'='*70}")
    
    gcs_path = f"gs://{bucket_name}/{input_file}"
    print(f"Reading from: {gcs_path}")
    
    df_original = spark.read.csv(gcs_path, header=True, inferSchema=True)
    
    print("Converting timestamp column...")
    df_original = df_original.withColumn(
        "Timestamp",
        F.to_timestamp(F.col("Timestamp"), "dd-MM-yyyy HH:mm")
    )
    
    print("Counting original rows...")
    row_count = df_original.count()
    print(f"✓ Original data loaded: {row_count:,} rows")
    
    print("\nComputing date range...")
    date_stats = df_original.agg(
        F.min("Timestamp").alias("min_date"),
        F.max("Timestamp").alias("max_date")
    ).collect()[0]
    print(f"✓ Date range: {date_stats['min_date']} to {date_stats['max_date']}")
    
    print("\nComputing missing values per column...")
    for col_name in df_original.columns:
        if col_name != "Timestamp":
            null_count = df_original.filter(F.col(col_name).isNull()).count()
            if null_count > 0:
                pct = (null_count / row_count) * 100
                print(f"  • {col_name}: {null_count:,} missing ({pct:.2f}%)")
    
    print(f"{'='*70}\n")
    return df_original

def fill_with_kaggle_and_forward(spark, df_original, df_kaggle):
    """Fill missing data using Kaggle data and forward fill"""
    print(f"\n{'='*70}")
    print("DATA FILLING PHASE")
    print(f"{'='*70}\n")
    
    print("JOB 12: Computing date range...")
    date_range = df_original.agg(
        F.min("Timestamp").alias("min_date"),
        F.max("Timestamp").alias("max_date")
    ).collect()[0]
    
    min_date = date_range['min_date']
    max_date = date_range['max_date']
    print(f"✓ Range: {min_date} to {max_date}")
    
    minutes_diff = int((max_date - min_date).total_seconds() / 60) + 1
    
    print(f"\nJOB 13: Creating full timestamp range ({minutes_diff:,} minutes)...")
    df_full_range = spark.range(minutes_diff) \
        .select(
            (F.expr(f"TIMESTAMP '{min_date}'") + F.expr("INTERVAL 1 MINUTE") * F.col("id")).alias("Timestamp")
        )
    
    total_timestamps = df_full_range.count()
    print(f"✓ Total timestamps created: {total_timestamps:,}")
    
    print("\nJOB 14-15: Performing left join with original data...")
    df_filled = df_full_range.join(df_original, "Timestamp", "left")
    
    print("Counting missing timestamps...")
    missing_count = df_filled.filter(F.col("Open").isNull()).count()
    print(f"✓ Missing timestamps found: {missing_count:,}")
    
    kaggle_cols = ["Open", "High", "Low", "Close", "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price"]
    
    print("\nJOB 16-17: Performing left join with Kaggle data...")
    df_with_kaggle = df_filled.join(
        df_kaggle.select("Timestamp", *kaggle_cols)
        .withColumnRenamed("Open", "K_Open")
        .withColumnRenamed("High", "K_High")
        .withColumnRenamed("Low", "K_Low")
        .withColumnRenamed("Close", "K_Close")
        .withColumnRenamed("Volume_(BTC)", "K_Volume_BTC")
        .withColumnRenamed("Volume_(Currency)", "K_Volume_Currency")
        .withColumnRenamed("Weighted_Price", "K_Weighted_Price"),
        "Timestamp",
        "left"
    )
    
    print("\nJOB 18: Filling nulls with Kaggle data using coalesce...")
    df_filled = df_with_kaggle \
        .withColumn("Open", F.coalesce(F.col("Open"), F.col("K_Open"))) \
        .withColumn("High", F.coalesce(F.col("High"), F.col("K_High"))) \
        .withColumn("Low", F.coalesce(F.col("Low"), F.col("K_Low"))) \
        .withColumn("Close", F.coalesce(F.col("Close"), F.col("K_Close"))) \
        .withColumn("Volume_(BTC)", F.coalesce(F.col("Volume_(BTC)"), F.col("K_Volume_BTC"))) \
        .withColumn("Volume_(Currency)", F.coalesce(F.col("Volume_(Currency)"), F.col("K_Volume_Currency"))) \
        .withColumn("Weighted_Price", F.coalesce(F.col("Weighted_Price"), F.col("K_Weighted_Price"))) \
        .drop("K_Open", "K_High", "K_Low", "K_Close", "K_Volume_BTC", "K_Volume_Currency", "K_Weighted_Price")
    
    print("Counting Kaggle-filled rows...")
    kaggle_filled = missing_count - df_filled.filter(F.col("Open").isNull()).count()
    print(f"✓ Filled from Kaggle: {kaggle_filled:,} rows")
    
    # Forward fill
    window_spec = Window.orderBy("Timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    
    price_columns = ['Open', 'High', 'Low', 'Close', 'Weighted_Price']
    
    print(f"\nJOB 19-23: Applying forward fill with window functions...")
    print(" This may take several minutes...")
    
    for i, col_name in enumerate(price_columns, 1):
        print(f"  Processing column '{col_name}' ({i}/{len(price_columns)})...")
        df_filled = df_filled.withColumn(
            col_name,
            F.last(col_name, ignorenulls=True).over(window_spec)
        )
    
    print("✓ Forward fill complete")
    
    print("\nJOB 24: Filling volume columns with 0...")
    df_filled = df_filled \
        .withColumn("Volume_(BTC)", F.coalesce(F.col("Volume_(BTC)"), F.lit(0))) \
        .withColumn("Volume_(Currency)", F.coalesce(F.col("Volume_(Currency)"), F.lit(0)))
    
    print("\nFinal count...")
    final_count = df_filled.count()
    print(f"✓ Final row count: {final_count:,}")
    
    forward_filled = missing_count - kaggle_filled
    
    stats = {
        'total_missing': missing_count,
        'kaggle_filled': kaggle_filled,
        'forward_filled': forward_filled,
        'zero_volume': missing_count - kaggle_filled
    }
    
    print("✓ DATA FILLING PHASE COMPLETE")
    
    return df_filled, stats

def process_bitcoin_data(bucket_name, input_blob, output_blob, kaggle_blob):
    """Main processing function"""
    print("Bitcoin Data Filler - Dataproc Edition")
    
    spark = create_spark_session()
    
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        kaggle_blob_obj = bucket.blob(kaggle_blob)
        
        if not kaggle_blob_obj.exists():
            print(f"\n ERROR: Kaggle file not found in GCS!")
            print(f"Expected: gs://{bucket_name}/{kaggle_blob}")
            print("\nPlease upload the Kaggle file:")
            print(f" gsutil cp btcusd_1-min_data.csv gs://{bucket_name}/")
            return None
        
        print(f"\n✓ Kaggle file found: gs://{bucket_name}/{kaggle_blob}")
        
        # Load data
        df_original = load_original_data(spark, bucket_name, input_blob)
        df_kaggle = load_kaggle_data(spark, bucket_name, kaggle_blob)
        
        # Fill missing data
        df_filled, stats = fill_with_kaggle_and_forward(spark, df_original, df_kaggle)
        
        print("FILLING STATISTICS")
        print(f"Total missing timestamps:        {stats['total_missing']:,}")
        print(f"Filled from Kaggle (volume > 0): {stats['kaggle_filled']:,} ({stats['kaggle_filled']/stats['total_missing']*100:.2f}%)")
        print(f"Filled by forward fill:          {stats['forward_filled']:,} ({stats['forward_filled']/stats['total_missing']*100:.2f}%)")
        print(f"Volume set to 0:                 {stats['zero_volume']:,}")
        
        # Check for remaining nulls
        print("\nJOB 25: Checking for remaining nulls...")
        all_complete = True
        for col_name in df_filled.columns:
            if col_name != "Timestamp":
                null_count = df_filled.filter(F.col(col_name).isNull()).count()
                if null_count > 0:
                    total_count = df_filled.count()
                    pct = (null_count / total_count) * 100
                    print(f"  {col_name}: {null_count:,} ({pct:.2f}%)")
                    all_complete = False
        
        if all_complete:
            print(" All columns complete!")
        
        # Final count
        print(f"\nFinal row count: {df_filled.count():,}")
        
        # Write output
        print(f"\n Writing to GCS...")
        output_path = f"gs://{bucket_name}/{output_blob}"
        
        # Write as single CSV file
        df_filled.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path + "_temp")
        
        print("Moving output file to final location...")
        subprocess.run([
            "bash", "-c",
            f"gsutil mv gs://{bucket_name}/{output_blob}_temp/part-*.csv gs://{bucket_name}/{output_blob} && gsutil rm -r gs://{bucket_name}/{output_blob}_temp"
        ], check=True)
        
        print(f"✓ Saved to: gs://{bucket_name}/{output_blob}")
        
        print("\n" + "="*70)
        print("✓ ALL JOBS COMPLETE!")
        print("✓ Check Dataproc Console for Spark UI")
        print("="*70)
        
        return df_filled
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()

if __name__ == "__main__":
    print("\nBitcoin Data Filler - Dataproc Edition")
    print("This script will:")
    print("1. Load original data from GCS")
    print("2. Load Kaggle data from GCS")
    print("3. Fill missing data with Kaggle data (volume > 0)")
    print("4. Use forward fill for remaining gaps")
    print("5. Write results back to GCS")
    print()
    
    try:
        result = process_bitcoin_data(
            bucket_name=BUCKET_NAME,
            input_blob=INPUT_FILE,
            output_blob=OUTPUT_FILE,
            kaggle_blob=KAGGLE_FILE
        )
        
        if result is not None:
            print("\n SUCCESS! Your data is ready.")
        else:
            print("\n FAILED. Check error messages above.")
        
    except Exception as e:
        print(f"\n FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        