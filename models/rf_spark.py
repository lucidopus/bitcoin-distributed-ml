"""
Data Preparation Script for BTC Price Prediction
Creates the Target variable and ensures proper data format
Run this locally before uploading to GCP
"""

import pandas as pd
import numpy as np
from datetime import datetime

def prepare_btc_data(input_file, output_file):
    """
    Prepare BTC data for prediction model
    
    Args:
        input_file: Path to raw CSV file
        output_file: Path to save prepared CSV
    """
    print("Loading data...")
    df = pd.read_csv(input_file)
    
    print(f"Initial shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Sort by timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    # Create target variable if not exists
    if 'Target' not in df.columns:
        print("\nCreating Target variable...")
        # Shift close price to get future price (15 minutes ahead)
        df['Future_Close'] = df['Close'].shift(-1)
        
        # Target: 1 if price goes up, 0 if down
        df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
        
        # Remove the last row (no future data)
        df = df[:-1]
        
        # Drop temporary column
        df = df.drop('Future_Close', axis=1)
    
    # Calculate additional features if missing
    if 'Feat_SMA_5' not in df.columns:
        print("\nCalculating Simple Moving Averages...")
        df['Feat_SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['Feat_SMA_10'] = df['Close'].rolling(window=10, min_periods=1).mean()
        df['Feat_SMA_15'] = df['Close'].rolling(window=15, min_periods=1).mean()
    
    if 'Feat_Vol_Std' not in df.columns:
        print("\nCalculating Volatility features...")
        df['Feat_Vol_Std'] = df['Close'].rolling(window=10, min_periods=1).std()
    
    if 'Feat_GK_Vol' not in df.columns:
        print("\nCalculating Garman-Klass Volatility...")
        # Simplified GK volatility
        hl = np.log(df['High'] / df['Low']) ** 2
        co = np.log(df['Close'] / df['Open']) ** 2
        df['Feat_GK_Vol'] = np.sqrt(0.5 * hl - (2 * np.log(2) - 1) * co)
    
    # Handle missing values
    print("\nHandling missing values...")
    print(f"Missing values before:\n{df.isnull().sum()}")
    
    # Forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Missing values after:\n{df.isnull().sum()}")
    
    # Ensure proper data types
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume_BTC', 
                      'Volume_Currency', 'Weighted_Price', 'Feat_SMA_5',
                      'Feat_SMA_10', 'Feat_SMA_15', 'Feat_GK_Vol', 'Feat_Vol_Std']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any remaining rows with NaN
    df = df.dropna()
    
    # Display statistics
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    print(f"Total records: {len(df)}")
    print(f"\nTarget distribution:")
    print(df['Target'].value_counts())
    print(f"\nTarget ratio:")
    print(df['Target'].value_counts(normalize=True))
    
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    print(df[numeric_columns + ['Target']].describe())
    
    # Save prepared data
    print(f"\nSaving prepared data to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"✓ Data saved successfully!")
    print(f"Final shape: {df.shape}")
    
    # Display sample
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    return df

def validate_data(df):
    """Validate data quality"""
    print("\n" + "="*60)
    print("DATA VALIDATION")
    print("="*60)
    
    issues = []
    
    # Check for infinite values
    inf_cols = df.columns[np.isinf(df.select_dtypes(include=[np.number])).any()]
    if len(inf_cols) > 0:
        issues.append(f"Infinite values in: {inf_cols.tolist()}")
    
    # Check for negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns and (df[col] <= 0).any():
            issues.append(f"Negative or zero values in: {col}")
    
    # Check price relationships
    if 'High' in df.columns and 'Low' in df.columns:
        if (df['High'] < df['Low']).any():
            issues.append("High < Low in some rows")
    
    # Check target balance
    if 'Target' in df.columns:
        target_ratio = df['Target'].mean()
        if target_ratio < 0.3 or target_ratio > 0.7:
            issues.append(f"Imbalanced target: {target_ratio:.2%} positive class")
    
    if issues:
        print("⚠ WARNINGS:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All validation checks passed!")
    
    return len(issues) == 0

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python prepare_data.py <input_csv> [output_csv]")
        print("\nExample:")
        print("  python prepare_data.py raw_btc_data.csv btc_data.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "btc_data_prepared.csv"
    
    try:
        df = prepare_btc_data(input_file, output_file)
        validate_data(df)
        
        print("\n" + "="*60)
        print("✓ DATA PREPARATION COMPLETE")
        print("="*60)
        print(f"\nYou can now upload {output_file} to GCP:")
        print(f"  gsutil cp {output_file} gs://YOUR_BUCKET/data/")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)