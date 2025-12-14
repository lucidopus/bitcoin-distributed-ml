# Project Plan: Distributed Bitcoin Trend Prediction Engine

## 1. Executive Summary
**Project Lead:** AI/ML Engineering Team  
**Technology Stack:** Apache Spark, PySpark (MLLib), Python  
**Objective:** To develop a scalable machine learning pipeline capable of predicting short-term Bitcoin price trends. The critical focus is not merely on model accuracy but on demonstrating distributed computing capabilities, efficient resource management, and scalability analysis across varying cluster configurations.

---

## 2. Phase I: Data Engineering & Ingestion
**Goal:** Establish a robust ETL pipeline to handle raw market data and prepare it for distributed processing.

* **Data Acquisition:** Ingestion of historical Bitcoin market data (Open, High, Low, Close, Volume) into the Spark environment.
* **Schema Definition:** Strict schema enforcement to ensure data type consistency (DoubleType, TimestampType) across distributed nodes.
* **Preprocessing:**
    * Handling missing values (Imputation vs. Dropping) ensuring time-series continuity.
    * Timestamp alignment and windowing preparations.
* **Artifacts:** Cleaned Spark DataFrames ready for transformation.

---

## 3. Phase II: Feature Engineering
**Goal:** Transform raw time-series data into predictive signals using distributed transformations.

* **Technical Indicator Generation:**
    * Implementation of sliding window operations to calculate indicators such as Moving Averages (SMA/EMA), RSI (Relative Strength Index), and Volatility metrics.
    * Derivation of the "Target" variable (e.g., Binary Classification: 1 if Price Up, 0 if Price Down).
* **Vectorization (Spark MLLib Requirement):**
    * Consolidation of numerical features into feature vectors using `VectorAssembler`.
* **Normalization:**
    * Application of `StandardScaler` or `MinMaxScaler` to normalize features, ensuring faster convergence for gradient-based algorithms.
* **Data Splitting:**
    * Chronological train-test splitting to prevent data leakage (respecting the time-series nature of the data).

---

## 4. Phase III: Model Training & Development
**Goal:** Train and tune three distinct distributed machine learning models to establish a performance baseline and a champion model.

We will leverage the following algorithms from the `pyspark.ml` library:

### Model A: Logistic Regression
* **Role:** Baseline Model.
* **Focus:** Linearity, interpretability, and training speed. Used to establish the "floor" for performance metrics.

### Model B: Random Forest Classifier
* **Role:** Ensemble Method (Bagging).
* **Focus:** Handling non-linear relationships and high-dimensionality. We will leverage Spark’s ability to parallelize tree construction across the cluster.

### Model C: Gradient Boosted Trees (GBT)
* **Role:** Ensemble Method (Boosting).
* **Focus:** High predictive accuracy. This tests the cluster's ability to handle sequential training optimizations compared to the parallel nature of Random Forest.

---

## 5. Phase IV: Cluster Management & Scalability Testing
**Goal:** Analyze how the system performs under constrained and expanded resources. This is the core systems engineering component of the project.

We will execute the training pipeline under the following configurations to test **Data Scalability**:

* **Configuration A (25% Data):** Baseline latency and resource consumption check.
* **Configuration B (50% Data):** Mid-load performance analysis.
* **Configuration C (75% Data):** Stress testing prior to full load.
* **Configuration D (100% Data):** Full-scale production simulation.

*Note: For each configuration, we will log training time, inference latency, and memory overhead.*

---

## 6. Phase V: Monitoring, Visualization & Evaluation
**Goal:** Quantify performance and visualize the behavior of the distributed system.

* **Model Evaluation Metrics:**
    * Calculation of Accuracy, Precision, Recall, and F1-Score.
    * Confusion Matrix generation for all three models.
* **System Monitoring:**
    * **Spark UI Analysis:** detailed review of DAG (Directed Acyclic Graph) visualization, Stage/Task duration, and Shuffle Read/Write sizes.
    * **Resource Visualization:** Plotting Training Time vs. Data Volume (Scalability Curves) for all three models.
* **Final Report Generation:** Synthesis of model performance against cluster resource utilization.

---

## 7. Definition of Done (DoD)
The project is considered complete when:
1.  [ ] All three models are successfully trained and saved.
2.  [ ] Scalability benchmarks (25% - 100%) are recorded and visualized.
3.  [ ] A comparative analysis identifies the optimal model based on the trade-off between Accuracy and Computational Cost.
4.  [ ] The codebase is modular, commented, and ready for hand-off.