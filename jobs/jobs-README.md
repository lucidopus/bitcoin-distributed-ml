# Job Scripts

This directory houses the PySpark scripts that form the core of our data processing and machine learning pipeline. The scripts were used along with the `infra` scripts to run on the Google Cloud Dataproc cluster. We maintained separate scripts for certain steps (Which usually go together as a unified script) to preserve a clear logical separation of concerns and to make step-by-step debugging easier.

## File Descriptions

- **`data-preprocessing.py`**: The first step in our pipeline. It merges our primary dataset with auxiliary Kaggle data to fill gaps and performs forward-filling to ensure a continuous time series.
- **`data-scaling.py`**: Handles data normalization by applying standard scaling to the feature set. This step is critical for the performance of models such as the Multilayer Perceptron. 
- **`eda.py`**: A specialized job for Exploratory Data Analysis. It computes statistics and generates visualizations to help us understand the data's underlying distributions and trends.
- **`feature_engineering/`**: Contains scripts dedicated to generating additive features, such as technical indicators, from the raw market data.
- **`training/`**: The home for our model training scripts. Each file here corresponds to a specific machine learning algorithm implemented in PySpark.
