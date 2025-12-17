# Job Scripts

This directory houses the PySpark scripts that form the core of our data processing and machine learning pipeline.

## File Descriptions

- **`data-preprocessing.py`**: The first step in our pipeline. It merges our primary dataset with auxiliary Kaggle data to fill gaps and performs forward-filling to ensure a continuous time series.
- **`data-scaling.py`**: Responsible for normalizing our data. It applies standard scaling to our features, which is crucial for the performance of models like the Multilayer Perceptron.
- **`eda.py`**: A specialized job for Exploratory Data Analysis. It computes statistics and generates visualizations to help us understand the data's underlying distributions and trends.
- **`feature_engineering/`**: Contains scripts dedicated to generating additive features, such as technical indicators, from the raw market data.
- **`training/`**: The home for our model training scripts. Each file here corresponds to a specific machine learning algorithm implemented in PySpark.
