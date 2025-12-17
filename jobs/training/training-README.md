# Model Training Scripts

This directory contains the implementations of the machine learning models we are evaluating. The models are implemented in Spark and are designed to run on a distributed computing environment.

## File Descriptions

- **`gradient-boosted-trees.py`**: Our implementation of the Gradient Boosted Trees (GBT) classifier. This model is often our top performer.
- **`mlp.py`**: The Multilayer Perceptron (Neural Network) classifier. We use this to see if deep learning architectures can capture complex non-linear patterns in the Bitcoin data.
- **`logistic_regression.py`**: This was just a recent experiment with the logistic regression model (not used in the final analysis).
- **`rf_spark.py`**: The Random Forest classifier implementation. It serves as a strong ensemble baseline to compare against the Gradient Boosted Trees.
