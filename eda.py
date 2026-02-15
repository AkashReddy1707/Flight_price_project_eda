import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def flight_price_eda(df, target_col="Price", result_dir="reports"):
    os.makedirs(result_dir, exist_ok=True)

    # Summary statistics
    summary = []
    
    summary.append(df.describe(include='all').T)
    summary.append(f"Data Shape : {df.shape}")
    summary.append(f"Missing Values : {df.isnull().sum()}")
    summary.append(f"Duplicate Values : {df.duplicated().sum()}")

    # save clean data 
    df_cleaned = df.drop_duplicates()
    df_cleaned.to_csv(os.path.join(result_dir, "cleaned_data.csv"), index=False)

    # Outlier detection
    outlier_data = pd.DataFrame()
    if target_col in df.columns:
        outliers, lower, upper = detect_outliers_iqr(df, target_col)
        outlier_data = outliers
        outlier_data.to_csv(os.path.join(result_dir, "outliers.csv"), index=False)

        # ✅ FIXED HERE
        summary.append(f"Outliers Detected in {target_col} : {len(outliers)}")
        summary.append(f"Outliers Lower Bound : {lower}")
        summary.append(f"Outliers Upper Bound : {upper}")

        # Boxplot for target variable
        plt.figure(figsize=(8,6))
        sns.boxplot(x=df[target_col])
        plt.title(f"Boxplot of {target_col}")
        plt.savefig(os.path.join(result_dir, f"boxplot_{target_col}.png"))
        plt.close()
    
    # Correlation Matrix
    num_cols = df.select_dtypes(include=["int64", "float64"])
    if num_cols.shape[1] > 1:
        corr = num_cols.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.savefig(os.path.join(result_dir, "correlation_matrix.png"))
        plt.close()

    # Save Summary report
    with open(os.path.join(result_dir, "summary_report.txt"), "w") as f:
        for item in summary:
            if isinstance(item, pd.DataFrame):
                f.write(item.to_string())
            else:
                f.write(str(item))
            f.write("\n\n")

    return {
        "summary": summary,
        "outliers": outlier_data,
        "cleaned_data": df_cleaned
    }
