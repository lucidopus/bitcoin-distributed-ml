import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os
import subprocess
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

REGION="us-central1"
ZONE="us-central1-a"
PROJECT_ID="bitcoin-trend-prediction1"
BUCKET_NAME="bitcoin-trend-prediction1-data"
spark = SparkSession.builder.appName("Bitcoin_EDA").getOrCreate()
INPUT_PATH = f"gs://{BUCKET_NAME}/bitcoin_data_feature_engineered.csv"
OUTPUT_IMAGE_DIR = f"gs://{BUCKET_NAME}/eda/"

def save_plot_to_gcs(fig, filename):
    local_path = f"/tmp/{filename}"
    print(f"Saving locally to {local_path}...")
    fig.savefig(local_path, bbox_inches='tight', dpi=150)
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
    df = df.withColumn("Year", F.year(F.col("Timestamp")))
    df = df.withColumn("Month", F.month(F.col("Timestamp")))

    print(f"Total rows: {df.count():,}")

    # ===============================
    # Plot 1: Garman-Klass Volatility
    # ===============================
    print("\n=== Plot 1: Garman-Klass Volatility Over Years ===")
    gk_vol_yearly = df.groupBy("Year").agg(
        F.avg("Feat_GK_Vol").alias("Avg_GK_Vol")
    ).orderBy("Year").toPandas()

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(
        gk_vol_yearly['Year'],
        gk_vol_yearly['Avg_GK_Vol'],
        marker='o',
        linewidth=3,
        markersize=10,
        label='Average GK Volatility'
    )
    ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Garman-Klass Volatility', fontsize=12, fontweight='bold')
    ax1.set_title('Bitcoin Intraday Volatility Over Years (Garman-Klass)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot_to_gcs(fig1, "01_gk_volatility_over_years.png")

    # ===============================
    # Plot 2: Volume BTC Over Years
    # ===============================
    print("\n=== Plot 2: Bitcoin Volume (BTC) Over Years ===")
    volume_yearly = df.groupBy("Year").agg(
        F.sum("Volume_(BTC)").alias("Total_Volume_BTC"),
        F.avg("Volume_(BTC)").alias("Avg_Volume_BTC")
    ).orderBy("Year").toPandas()

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 10))

    ax2a.plot(
        volume_yearly['Year'],
        volume_yearly['Total_Volume_BTC'],
        marker='o',
        linewidth=2,
        markersize=8,
        label='Total Volume (BTC)'
    )
    ax2a.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2a.set_ylabel('Total Volume (BTC)', fontsize=12, fontweight='bold')
    ax2a.set_title('Total Bitcoin Trading Volume by Year', fontsize=14, fontweight='bold')
    ax2a.legend(fontsize=10)
    ax2a.grid(True, alpha=0.3)
    ax2a.ticklabel_format(style='plain', axis='y')

    ax2b.plot(
        volume_yearly['Year'],
        volume_yearly['Avg_Volume_BTC'],
        marker='s',
        linewidth=2,
        markersize=8,
        label='Average Volume (BTC)'
    )
    ax2b.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2b.set_ylabel('Average Volume (BTC)', fontsize=12, fontweight='bold')
    ax2b.set_title('Average Bitcoin Trading Volume by Year', fontsize=14, fontweight='bold')
    ax2b.legend(fontsize=10)
    ax2b.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot_to_gcs(fig2, "02_volume_btc_over_years.png")

    # ===============================
    # Plot 3A: Open Prices Over Years
    # ===============================
    print("\n=== Plot 3A: Average Open Prices Over Years ===")
    open_yearly = df.groupBy("Year").agg(
        F.avg("Open").alias("Avg_Open"),
        F.min("Open").alias("Min_Open"),
        F.max("Open").alias("Max_Open")
    ).orderBy("Year").toPandas()

    fig3a, ax3a = plt.subplots(figsize=(14, 6))
    ax3a.plot(
        open_yearly['Year'],
        open_yearly['Avg_Open'],
        marker='o',
        linewidth=2,
        markersize=8,
        label='Average Open Price'
    )
    ax3a.fill_between(
        open_yearly['Year'],
        open_yearly['Min_Open'],
        open_yearly['Max_Open'],
        alpha=0.2,
        label='Open Price Range'
    )

    ax3a.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3a.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax3a.set_title('Bitcoin Average Open Prices Over Years', fontsize=14, fontweight='bold')
    ax3a.legend(fontsize=10)
    ax3a.grid(True, alpha=0.3)
    ax3a.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    save_plot_to_gcs(fig3a, "03a_open_price_over_years.png")

    # ===============================
    # Plot 3B: Close Prices Over Years
    # ===============================
    print("\n=== Plot 3B: Average Close Prices Over Years ===")
    close_yearly = df.groupBy("Year").agg(
        F.avg("Close").alias("Avg_Close"),
        F.min("Close").alias("Min_Close"),
        F.max("Close").alias("Max_Close")
    ).orderBy("Year").toPandas()

    fig3b, ax3b = plt.subplots(figsize=(14, 6))
    ax3b.plot(
        close_yearly['Year'],
        close_yearly['Avg_Close'],
        marker='s',
        linewidth=2,
        markersize=8,
        label='Average Close Price'
    )
    ax3b.fill_between(
        close_yearly['Year'],
        close_yearly['Min_Close'],
        close_yearly['Max_Close'],
        alpha=0.2,
        label='Close Price Range'
    )

    ax3b.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax3b.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
    ax3b.set_title('Bitcoin Average Close Prices Over Years', fontsize=14, fontweight='bold')
    ax3b.legend(fontsize=10)
    ax3b.grid(True, alpha=0.3)
    ax3b.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    save_plot_to_gcs(fig3b, "03b_close_price_over_years.png")

    print("\n=== Plot 4: Feature Correlation Heatmap ===")

    heatmap_features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume_(BTC)",
    "Volume_(Currency)",
    "Weighted_Price",
    "Feat_SMA_5", "Feat_SMA_10", "Feat_SMA_15", 
    "Feat_GK_Vol", "Feat_Vol_Std",
    "Target"
    ]

    # Keep only existing columns (defensive)
    heatmap_features = [c for c in heatmap_features if c in df.columns]

    print(f"Using {len(heatmap_features)} features for correlation heatmap")

    # Sample aggressively to protect driver
    corr_pdf = (
        df.select(heatmap_features)
        .dropna()
        .sample(fraction=0.05, seed=42)
        .limit(50_000)   # HARD CAP
        .toPandas()
    )

    corr_matrix = corr_pdf.corr(method="pearson")

    fig4, ax4 = plt.subplots(figsize=(12, 10))
    im = ax4.imshow(corr_matrix.values)

    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=9)
    ax4.set_yticklabels(corr_matrix.columns, fontsize=9)
    ax4.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_plot_to_gcs(fig4, "04_feature_correlation_heatmap.png")

    print("\n--- EDA JOB COMPLETE ---")
    print(f"All plots saved to: {OUTPUT_IMAGE_DIR}")

    spark.stop()

if __name__ == "__main__":
    main()
