# Creating a new Jupyter notebook for bivariate analysis as per the user's request
# Re-importing necessary libraries and re-running the notebook creation code after reset
import nbformat as nbf

# Define the notebook cells for bivariate analysis
cells = [
    # Import necessary libraries and load data
    nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
                         
# Configure plot settings
%matplotlib inline
sns.set_style('whitegrid')  # Changed from plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [10, 6]

# Load the dataset
data_path = 'C:/Users/Denny/insurance-risk-modeling/data/processed/cleaned_insurance_data.csv'
df = pd.read_csv(data_path)
                
# Convert categorical columns to category type
categorical_columns = ['Gender', 'Driving_Experience', 'Vehicle_Type', 'Region', 'Accident']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Display first few rows
df.head()
"""),

    # Analyze relationships with target variable using boxplots and violinplots for numerical variables
    nbf.v4.new_markdown_cell("## Relationship Analysis with Target Variable (Accident)"),
    nbf.v4.new_code_cell("""
# Defining the target variable
target_var = 'Accident'

# Numerical columns for analysis (excluding Accident since it's categorical)
numerical_columns = ['Age', 'Previous_Accidents', 'Annual_Mileage', 'Premium']

# Boxplots and Violinplots for numerical columns against the target variable
for col in numerical_columns:
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(x=target_var, y=col, data=df)
    plt.title(f'Boxplot of {col} by {target_var}')
    plt.xlabel(target_var)
    plt.ylabel(col)

    plt.subplot(1, 2, 2)
    sns.violinplot(x=target_var, y=col, data=df)
    plt.title(f'Violin Plot of {col} by {target_var}')
    plt.xlabel(target_var)
    plt.ylabel(col)
    
    plt.tight_layout()
    plt.show()
"""),

    # Bar plots for categorical variables against target variable
    nbf.v4.new_markdown_cell("## Bar Plots for Categorical Variables by Target Variable"),
    nbf.v4.new_code_cell("""
# Categorical columns for analysis (excluding Accident since it's the target)
categorical_columns = ['Gender', 'Driving_Experience', 'Vehicle_Type', 'Region']
target_var = 'Accident'

for col in categorical_columns:
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x=col, hue=target_var)
    
    # Rotate x-labels if needed
    plt.xticks(rotation=45)
    
    # Add title and labels
    plt.title(f'Distribution of {col} by {target_var}')
    plt.xlabel(col)
    plt.ylabel('Count')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container)
    
    plt.tight_layout()
    plt.show()
"""),

    # Crosstabulation analysis for categorical variables
    nbf.v4.new_markdown_cell("## Crosstabulation Analysis for Categorical Variables"),
    nbf.v4.new_code_cell("""
# Crosstabs for each categorical variable with the target variable
for col in categorical_columns:
    print(f"\\nCrosstab for {col} vs {target_var}:\\n")
    
    # Raw counts
    ct_counts = pd.crosstab(df[col], df[target_var])
    print("Counts:")
    print(ct_counts)
    print("\\n")
    
    # Percentages
    ct_pct = pd.crosstab(df[col], df[target_var], normalize='index') * 100
    print("Percentages (%):")
    print(ct_pct.round(2))
    print("\\n" + "="*50)
"""),

    # Calculate and display the correlation matrix
    nbf.v4.new_markdown_cell("## Correlation Analysis"),
    nbf.v4.new_code_cell("""
# Calculate correlation matrix for numerical variables
correlation_matrix = df.corr()

# Display correlation matrix
correlation_matrix
"""),

    # Plotting the correlation matrix using a heatmap
    nbf.v4.new_markdown_cell("## Heatmap of Correlation Matrix"),
    nbf.v4.new_code_cell("""
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", square=True, cbar_kws={'shrink': .8})
plt.title('Correlation Matrix Heatmap')
plt.show()
""")

]

# Create the notebook
notebook = nbf.v4.new_notebook(cells=cells)

# Define the path for saving the notebook
#os.makedirs("notebooks", exist_ok=True)
notebook_path = "notebooks/03_bivariate_analysis.ipynb"

# Write the notebook file
with open(notebook_path, "w", encoding='utf-8') as f:
    nbf.write(notebook, f)