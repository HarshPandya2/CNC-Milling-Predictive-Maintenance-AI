# --- Hyperparameter Tuning with Optuna & Model Saving ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import pickle

# Load the dataset
file_path = "cleaned_mill.csv"
df = pd.read_csv(file_path)

# Handle missing values
df["VB"].interpolate(method='linear', inplace=True)
df.dropna(subset=["VB"], inplace=True)

# Outlier Detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(df)
df = df[outlier_labels == 1]

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df.drop(columns=["VB"]))
poly_feature_names = poly.get_feature_names_out(df.columns.drop("VB"))
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

df_poly["VB"] = np.log1p(df["VB"].values)  # Log transformation

# Prepare Data
X = df_poly.drop(columns=["VB"])
y = df_poly["VB"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optuna Hyperparameter Tuning
def objective(trial):
    param = {
        'lgbm': {
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3),
        },
        'xgb': {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
        },
        'cat': {
            'iterations': trial.suggest_int('cat_iterations', 100, 1000),
            'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.3),
        }
    }

    base_models = [
        ('lightgbm', lgb.LGBMRegressor(**param['lgbm'])),
        ('xgboost', xgb.XGBRegressor(**param['xgb'])),
        ('catboost', cb.CatBoostRegressor(**param['cat'], verbose=0))
    ]

    final_estimator = lgb.LGBMRegressor(n_estimators=500, random_state=42)
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
    stacking_model.fit(X_train, y_train)

    y_pred = stacking_model.predict(X_test)
    y_pred_actual = np.expm1(y_pred)
    y_test_actual = np.expm1(y_test)

    return mean_squared_error(y_test_actual, y_pred_actual)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300)

# Best Hyperparameters
print("Best Hyperparameters:", study.best_params)

# Final Training with Best Hyperparameters
best_params = study.best_params

base_models = [
    ('lightgbm', lgb.LGBMRegressor(
        n_estimators=best_params['lgbm_n_estimators'],
        learning_rate=best_params['lgbm_learning_rate']
    )),
    ('xgboost', xgb.XGBRegressor(
        n_estimators=best_params['xgb_n_estimators'],
        learning_rate=best_params['xgb_learning_rate']
    )),
    ('catboost', cb.CatBoostRegressor(
        iterations=best_params['cat_iterations'],
        learning_rate=best_params['cat_learning_rate'],
        verbose=0
    ))
]

final_estimator = lgb.LGBMRegressor(n_estimators=500, random_state=42)
stacking_model = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
stacking_model.fit(X_train, y_train)

# Predictions & Performance Evaluation
y_pred = stacking_model.predict(X_test)
y_pred_actual_scale = np.expm1(y_pred)
y_test_actual_scale = np.expm1(y_test)

mse = mean_squared_error(y_test_actual_scale, y_pred_actual_scale)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual_scale, y_pred_actual_scale)
r2 = r2_score(y_test_actual_scale, y_pred_actual_scale)

print("Hyper-Tuned Stacking Regressor Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Save the Model
model_path = "stacking_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(stacking_model, f)

print("Model saved successfully at:", model_path)
        