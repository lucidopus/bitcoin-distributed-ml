import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BUCKET_NAME = os.environ.get("BUCKET_NAME")

if not BUCKET_NAME:
    raise ValueError("Environment variable BUCKET_NAME is missing.")

INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_feature_engineered.csv"
OUTPUT_IMAGE_DIR = f"gs://{BUCKET_NAME}/eda/"

def save_plot_to_gcs(fig, filename):
    """
    Saves a Matplotlib figure locally to the worker node, 
    then uploads it to GCS using gsutil.
    """
    local_path = f"/tmp/{filename}"
    print(f"Saving locally to {local_path}...")
    fig.savefig(local_path, bbox_inches='tight')
    plt.close(fig)
    
    gcs_path = f"{OUTPUT_IMAGE_DIR}{filename}"
    print(f"Uploading to {gcs_path}...")
    
    try:
        subprocess.check_call(["gsutil", "cp", local_path, gcs_path])
        print(f"Successfully uploaded {filename}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR uploading {filename}: {e}")
    
    if os.path.exists(local_path):
        os.remove(local_path)

def main():
    spark = SparkSession.builder.appName("Bitcoin_EDA_Job").getOrCreate()
    
    print(f"--- STARTING EDA JOB ---")
    print(f"Reading Data from: {INPUT_PATH}")
    print(f"Saving Images to: {OUTPUT_IMAGE_DIR}")

    df = spark.read.csv(INPUT_PATH, header=True, inferSchema=True)
    df = df.withColumn("Timestamp", F.to_timestamp(F.col("Timestamp")))

    print("Generating Plot 1: Class Balance...")
    class_counts = df.groupBy("Target").count().toPandas()
    
    fig1 = plt.figure(figsize=(8, 6))
    sns.barplot(data=class_counts, x="Target", y="count", hue="Target", palette="viridis", legend=False)
    plt.title("Target Class Distribution (0=Down, 1=Up)")
    plt.xlabel("Target Class")
    plt.ylabel("Count")
    
    # Add count labels on top of bars
    for index, row in class_counts.iterrows():
        plt.text(row.name, row['count'], str(row['count']), color='black', ha="center")
        
    save_plot_to_gcs(fig1, "01_class_balance.png")

    print("Generating Plot 2: Correlation Matrix...")
    feature_cols = [
        "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
        "Feat_GK_Vol", "Feat_Vol_Std", 
        "Volume", "Target"
    ]
    
    # Select specific columns and sample
    pdf_corr = df.select(feature_cols).sample(fraction=0.1, seed=42).toPandas()
    
    fig2 = plt.figure(figsize=(10, 8))
    sns.heatmap(pdf_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    save_plot_to_gcs(fig2, "02_correlation_matrix.png")

    print("Generating Plot 3: Volatility Analysis...")
    recent_data = df.orderBy(F.col("Timestamp").desc()).limit(500).toPandas()
    recent_data = recent_data.sort_values("Timestamp")
    
    fig3, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Bitcoin Price (Close)', color=color)
    ax1.plot(recent_data['Timestamp'], recent_data['Close'], color=color, label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Garman-Klass Volatility', color=color)
    ax2.plot(recent_data['Timestamp'], recent_data['Feat_GK_Vol'], color=color, linestyle='--', alpha=0.5, label='GK Volatility')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Bitcoin Price vs. Intraday Volatility (Last 500 Mins)")
    save_plot_to_gcs(fig3, "03_volatility_analysis.png")

    print("--- EDA JOB COMPLETE ---")
    spark.stop()

if __name__ == "__main__":
    main()