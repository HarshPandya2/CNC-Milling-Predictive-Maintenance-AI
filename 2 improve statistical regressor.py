# --- Advanced Stacking Regressor with Hyperparameter Tuning ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
file_path = "cleaned_mill.csv"
df = pd.read_csv(file_path)

# Outlier Removal using Isolation Forest
iso_forest = IsolationForest(contamination=0.03, random_state=42)
outlier_labels = iso_forest.fit_predict(df.drop(columns=["VB"]))
df = df[outlier_labels == 1]

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df.drop(columns=["VB"]))
poly_feature_names = poly.get_feature_names_out(df.columns.drop("VB"))
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# Log Transformation
df_poly["VB"] = np.log1p(df["VB"].values)

# Prepare data
X = df_poly.drop(columns=["VB"])
y = df_poly["VB"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15]
}
param_grid_svr = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']
}
param_grid_knn = {
    'n_neighbors': [5, 10, 15]
}

# Grid Search
svr = GridSearchCV(SVR(), param_grid_svr, cv=3, scoring='r2', verbose=1)
svr.fit(X_train, y_train)

rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='r2', verbose=1)
rf.fit(X_train, y_train)

knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=3, scoring='r2', verbose=1)
knn.fit(X_train, y_train)

# Advanced Stacking Regressor
base_models = [
    ('svr', svr.best_estimator_),
    ('random_forest', rf.best_estimator_),
    ('knn', knn.best_estimator_),
    ('lightgbm', lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03)),
    ('xgboost', xgb.XGBRegressor(n_estimators=500, learning_rate=0.03)),
    ('catboost', cb.CatBoostRegressor(n_estimators=500, learning_rate=0.03, verbose=0))
]

final_estimator = GradientBoostingRegressor(n_estimators=1000, random_state=42)

stacking_model = StackingRegressor(estimators=base_models, final_estimator=final_estimator)
stacking_model.fit(X_train, y_train)

# Prediction & Performance Evaluation
y_pred = stacking_model.predict(X_test)
y_pred_actual_scale = np.expm1(y_pred)
y_test_actual_scale = np.expm1(y_test)

mse = mean_squared_error(y_test_actual_scale, y_pred_actual_scale)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_actual_scale, y_pred_actual_scale)
r2 = r2_score(y_test_actual_scale, y_pred_actual_scale)

print("✅ Advanced Stacking Regressor Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
