import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load dataset (Replace with actual dataset)
df = pd.read_csv("C:/Users/Rober/OneDrive/Documents/Research campus pc/Research/DRP/MERRA-2/Toolik Lake 1999_2020_test.csv")

# Convert inf values to NaN before any processing
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop any remaining NaN values
if df.isnull().values.any():
    print("\nWarning: Missing values detected. Dropping missing values.")
    df.dropna(inplace=True)

# Summary Statistics
print("\nBasic Descriptive Statistics:")
print(df.describe())

# Checking missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Mean, Median, Mode, Standard Deviation
for col in df.select_dtypes(include=[np.number]):
    if df[col].isnull().all():
        print(f"Skipping {col} due to all values being NaN.")
        continue
    print(f"\nStatistics for {col}:")
    print(f"Mean: {df[col].mean()}")
    print(f"Median: {df[col].median()}")
    print(f"Mode: {df[col].mode()[0]}")
    print(f"Standard Deviation: {df[col].std()}")
    print(f"Skewness: {df[col].skew()}")
    print(f"Kurtosis: {df[col].kurt()}")

    # Histogram and KDE plot
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

    # Probability Plot
    stats.probplot(df[col].dropna(), dist="norm", plot=plt)
    plt.title(f"Probability Plot for {col}")
    plt.show()

# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Detecting Outliers using IQR method
for col in df.select_dtypes(include=[np.number]):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"\nOutliers in {col}: {len(outliers)}")

# Detecting Outliers using Z-score
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outlier_indices = np.where(z_scores > 3)
print(f"\nTotal Outliers (Z-score > 3): {len(outlier_indices[0])}")


# Pairplot for variable relationships --> only use if there are a couple of variables!
#sns.pairplot(df.select_dtypes(include=[np.number]))
#plt.show()

print("EDA Completed.")