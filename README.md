# Prediction of Tool Wear During Milling Operation in EN8 Using AI Techniques

An end-to-end machine learning pipeline for condition monitoring, flank wear estimation, and surface roughness prediction in CNC face milling using vibration signals and stacking regression architectures.

---

## Overview

In milling operations, cutting tool degradation directly impacts surface finish, dimensional tolerances, and machine downtime. This project implements a data-driven predictive maintenance framework focused on the face milling of EN8 medium carbon steel.

By acquiring dynamic triaxial acceleration data during cutting passes, the pipeline extracts statistical and probabilistic features, trains multiple baseline regression models, and uses a stacked ensemble model to predict:
1. Tool Flank Wear (VB)
2. Surface Roughness (Ra)

The framework was evaluated on an experimental laboratory setup and validated using the standard PHM 2010 milling benchmark dataset.

---

## Experimental Setup

* Workpiece Material: EN8 medium carbon steel (0.36–0.44% C, 180–230 HB)
* CNC Machine: BMV 35 Gaurav CNC Milling Machine (spindle power 5.5 kW, max 8000 RPM)
* Cutting Tool: Mitsubishi ASX445 face mill cutter (50 mm diameter, 4 flutes) with SEMT13T3AGSN-JM VP15TF PVD-coated carbide inserts
* Cutting Conditions:
  * Spindle Speed: 1200 RPM
  * Feed Rate: 100 mm/min
  * Axial Depth of Cut: 0.08 mm
* Sensors and Metrology:
  * Vibration: Kistler triaxial accelerometer mounted on the tool arbor (sampling rate: 2000 Hz)
  * Flank Wear: AROS Vision Measuring Machine (optical VMM)
  * Surface Roughness: Mitutoyo Contact-Type Surface Roughness Tester (Ra)

---

## Machine Learning Pipeline

### 1. Data Preprocessing and Cleaning
* Outlier rejection using the Interquartile Range (IQR) technique and Isolation Forest algorithms.
* Linear interpolation across passes for wear progression tracking.
* Missing value filtering and standardization using scikit-learn's StandardScaler.

### 2. Feature Engineering
* Time-Series Segmentation: Sliding window approach (2000 data points per window, 500-point step size).
* Statistical Domain Metrics: Extracted mean, standard deviation, root mean square (RMS), kurtosis, and skewness across all three axes (X, Y, Z) to capture signal amplitude, energy, and distribution shape (15 features).
* Probabilistic Latent States: A 4-component Gaussian Mixture Model (GMM) was fitted to the multi-axis accelerometer signals to approximate operational Hidden Markov Model (HMM) states, capturing structural operational changes within the cut (4 features).
* Total Feature Vector: 19 vibration-derived features per window.

### 3. Ensemble Architecture
The core model uses a multi-layer Stacking Regressor:
* Base Estimators:
  * Support Vector Regression (SVR with RBF kernel)
  * Random Forest Regressor
  * LightGBM Regressor
  * XGBoost Regressor
  * CatBoost Regressor
* Meta-Learner: Gradient Boosting Regressor trained on the out-of-fold predictions of the base estimators.
* Optimization: Hyperparameter tuning via Optuna and Grid Search to minimize mean squared error.
* Statistical Validation: Residual checks, Shapiro-Wilk normality tests, and Variance Inflation Factor (VIF) evaluation.

---

## Results and Metrics

| Dataset | Target Parameter | Model Architecture | R² Score | MSE | RMSE | MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Experimental (EN8) | Tool Wear (VB) | Stacking Regressor | 0.9992 | 2e-05 | 0.0004 mm | 0.0003 mm |
| Experimental (EN8) | Surface Roughness (Ra) | Stacking Regressor | 0.9993 | 0.0004 | 0.0204 µm | 0.0152 µm |
| PHM 2010 Benchmark | Tool Wear (VB) | Stacking Regressor | 0.9885 | 9.8171 | 3.1332 | 1.5320 |

---

## Repository Structure

```text
├── 1_Exploratory_Data_Analysis.py             # Feature distribution and correlation analysis
├── 2_Data_Preprocessing.py                     # Outlier cleaning (IQR, Isolation Forest) and scaling
├── 3_Statistical_Validation.py                 # Residual plots, Shapiro-Wilk test, and VIF check
├── 4_Optuna_Hyperparameter_Tuning.py           # Bayesian optimization of ensemble models
├── 5_Final_Stacking_Model.py                   # Production stacking model training and evaluation
├── mill.csv                                    # Raw milling data
├── cleaned_mill.csv                            # Cleaned and processed dataset
├── LICENSE                                     # MIT License
└── README.md                                   # Project documentation
