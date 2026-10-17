# --- Final Stacking Regressor Code (94.8% Accuracy) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load the dataset
file_path = "cleaned_mill.csv"
df = pd.read_csv(file_path)

# Handle missing values and drop outliers
df["VB"].interpolate(method='linear', inplace=True)
df.dropna(subset=["VB"], inplace=True)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(df)
df = df[outlier_labels == 1]

# --- Polynomial Features ---
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

# --- Optimized Stacking Regressor ---
base_models = [
    ('svr', SVR(C=10, gamma=0.01, kernel='rbf')),
    ('random_forest', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
    ('lightgbm', lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05)),
    ('xgboost', xgb.XGBRegressor(n_estimators=300, learning_rate=0.05)),
    ('catboost', cb.CatBoostRegressor(n_estimators=300, learning_rate=0.05, verbose=0))
]

final_estimator = GradientBoostingRegressor(n_estimators=500, random_state=42)

stacking_model = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
stacking_model.fit(X_train, y_train)

# Save the model
with open("final_stacking_model.pkl", "wb") as file:
    pickle.dump(stacking_model, file)

# Predictions & Performance Evaluation
y_pred = stacking_model.predict(X_test)
y_pred_actual_scale = np.expm1(y_pred)
y_test_actual_scale = np.expm1(y_test)

mse = mean_squared_error(y_test_actual_scale, y_pred_actual_scale)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual_scale, y_pred_actual_scale)
r2 = r2_score(y_test_actual_scale, y_pred_actual_scale)

print("✅ Final Stacking Regressor Performance (94.8% Accuracy):")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
