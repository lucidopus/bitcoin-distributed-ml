# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `infra/create_cluster.sh` for provisioning Dataproc clusters
- Added `infra/delete_cluster.sh` for tearing down Dataproc clusters
- Added `jobs/feature-engineering.py` for PySpark-based feature engineering
- Added `requirements.txt` specifically for the project dependencies
- Updated `.gitignore` with standard Python and OS exclusions
- Added data preprocessing script (`jobs/data-preprocessing.py`) for filling missing Bitcoin data using PySpark, integrating Kaggle data and forward fill on Dataproc

### Changed

### Deprecated

### Removed

### Fixed

### Security