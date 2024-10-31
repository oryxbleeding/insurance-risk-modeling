# Re-importing necessary libraries and re-running the notebook creation code after reset
import nbformat as nbf
import os

# Define the notebook cells for univariate analysis
cells = [
    # Import necessary libraries
    nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = 'C:/Users/Denny/insurance-risk-modeling/data/processed/cleaned_insurance_data.csv'
df = pd.read_csv(data_path)

# Display first few rows
df.head()
"""),

    # Statistical summary for numerical columns
    nbf.v4.new_markdown_cell("## Statistical Summary for Numerical Variables"),
    nbf.v4.new_code_cell("""
# Statistical summary
df.describe()
"""),

    # Histograms and density plots for numerical variables
    nbf.v4.new_markdown_cell("## Histograms and Density Plots for Numerical Variables"),
    nbf.v4.new_code_cell("""
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_columns:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, bins=20, edgecolor='k')
    plt.title(f'Histogram and Density Plot of {col}')
    plt.xlabel(col)

    plt.subplot(1, 2, 2)
    sns.kdeplot(df[col], fill=True)
    plt.title(f'Density Plot of {col}')
    plt.xlabel(col)
    
    plt.tight_layout()
    plt.show()
"""),

    # Bar plots for categorical variables
    nbf.v4.new_markdown_cell("## Frequency Distributions for Categorical Variables"),
    nbf.v4.new_code_cell("""
categorical_columns = df.select_dtypes(include=['category', 'object']).columns

for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=df[col], edgecolor='k')
    plt.title(f'Bar Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.show()
"""),

    # Additional statistical analysis (mean, median, mode, std)
    nbf.v4.new_markdown_cell("## Additional Statistical Analysis"),
    nbf.v4.new_code_cell("""
# Mean, median, mode, and standard deviation
for col in numerical_columns:
    mean = df[col].mean()
    median = df[col].median()
    mode = df[col].mode().iloc[0]
    std_dev = df[col].std()
    
    print(f"{col}:")
    print(f"  Mean: {mean}")
    print(f"  Median: {median}")
    print(f"  Mode: {mode}")
    print(f"  Standard Deviation: {std_dev}\\n")
""")
]

# Create the notebook
notebook = nbf.v4.new_notebook(cells=cells)

# Define the path for saving the notebook
#os.makedirs("notebooks", exist_ok=True)
notebook_path = "notebooks/02_univariate_analysis.ipynb"

# Write the notebook file
with open(notebook_path, "w", encoding='utf-8') as f:
    nbf.write(notebook, f)
