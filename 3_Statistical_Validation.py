import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import make_scorer
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import qqplot
from statsmodels.stats.stattools import durbin_watson
import numpy as np

# Load the dataset
file_path = "mill.csv"
df = pd.read_csv(file_path)

# Drop unnecessary column
df.drop(columns=["Unnamed: 0"], inplace=True)

# Handle missing values in VB (flank wear)
df["VB"].interpolate(method='linear', inplace=True)  # Linear interpolation for missing values

# Check data distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["VB"], bins=20, kde=True)
plt.title("Distribution of Flank Wear (VB)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Scatter plot to show data spread
plt.figure(figsize=(10, 6))
sns.pairplot(df, diag_kind='kde')
plt.suptitle("Pairwise Feature Distribution", y=1.02)
plt.show()

# Prepare data for regression
X = df.drop(columns=["VB"])  # Features
y = df["VB"]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define base models
base_models = [
    ('svr', SVR(C=1, gamma=0.1, kernel='rbf')),
    ('decision_tree', DecisionTreeRegressor(max_depth=3, min_samples_split=10, random_state=42)),
    ('random_forest', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
]

# Stacking Regressor
stacking_model = StackingRegressor(estimators=base_models, final_estimator=RandomForestRegressor(n_estimators=100, random_state=42))
stacking_model.fit(X_train, y_train)

# Prediction and performance metrics
y_pred = stacking_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Stacking Regressor Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# --- Statistical Validation ---
# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

# Normality Check
qqplot(residuals, line='s')
plt.title("Q-Q Plot for Residuals")
plt.show()

shapiro_test = shapiro(residuals)
print(f"Shapiro-Wilk Test: p-value = {shapiro_test.pvalue:.4f}")

# Homoscedasticity Check
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, np.abs(residuals))
plt.axhline(y=np.mean(np.abs(residuals)), color='r', linestyle='--')
plt.title("Homoscedasticity Check")
plt.xlabel("Predicted Values")
plt.ylabel("|Residuals|")
plt.show()

# Multicollinearity Check (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# Autocorrelation Check
dw_test = durbin_watson(residuals)
print(f"Durbin-Watson Test: {dw_test:.4f}")

# Export to Excel
output_data = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R2 Score', 'Shapiro-Wilk p-value', 'Durbin-Watson Test'],
    'Value': [mse, rmse, mae, r2, shapiro_test.pvalue, dw_test]
})

with pd.ExcelWriter("Statistical_Validation.xlsx") as writer:
    output_data.to_excel(writer, sheet_name="Performance", index=False)
    vif_data.to_excel(writer, sheet_name="VIF", index=False)

print("Exported statistical validation to Excel.")