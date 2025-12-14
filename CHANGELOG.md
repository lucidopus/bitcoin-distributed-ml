# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `infra/create_cluster.sh` for provisioning Dataproc clusters
- Added `infra/delete_cluster.sh` for tearing down Dataproc clusters
- Added `infra/submit-job.sh` for submitting PySpark jobs to Dataproc clusters
- Added `jobs/feature-engineering.py` for PySpark-based feature engineering
- Added `requirements.txt` specifically for the project dependencies
- Updated `.gitignore` with standard Python and OS exclusions
- Added data preprocessing script (`jobs/data-preprocessing.py`) for filling missing Bitcoin data using PySpark, integrating Kaggle data and forward fill on Dataproc
- Added `.env.example` with placeholder environment variables for GCP configuration
- Added `jobs/eda.py` for performing Exploratory Data Analysis (EDA) on Dataproc, generating and uploading plots (Class Balance, Correlation Matrix, Volatility Analysis) to GCS

### Changed

- Updated `infra/create_cluster.sh` to enforce mandatory `--workers` argument and removed `--cluster-name` override (now strictly uses .env)
- Updated `README.md` with comprehensive project documentation, including architecture, directory structure, and usage instructions
- Refactored `infra/create_cluster.sh` and `infra/delete_cluster.sh` to load environment variables from `.env` file instead of using hardcoded values
- Refactored `jobs/feature-engineering.py` and `jobs/data-preprocessing.py` to use environment variables (`PROJECT_ID`, `BUCKET_NAME`) instead of hardcoded values
### Deprecated

### Removed

### Fixed

### Security