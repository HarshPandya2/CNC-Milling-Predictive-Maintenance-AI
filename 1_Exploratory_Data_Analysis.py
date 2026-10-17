import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Display cleaned data info
df.info()
print(df.head())
