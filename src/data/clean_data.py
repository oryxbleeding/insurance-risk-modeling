
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
from pathlib import Path

def clean_data(input_path='insurance_data.csv', output_path='cleaned_insurance_data.csv'):
    # Get the absolute path relative to this script
    script_dir = Path(__file__).parent.parent.parent  # Go up three levels from src/data to root
    input_path = script_dir / 'data' / 'raw' / input_path
    output_path = script_dir / 'data' / 'processed' / output_path
    
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Define numerical and categorical columns
    numerical_columns = ['Age', 'Previous_Accidents', 'Annual_Mileage', 'Premium']
    categorical_columns = ['Gender', 'Driving_Experience', 'Vehicle_Type', 'Region', 'Accident']
    
    # Identifying outliers using Z-score
    outlier_threshold = 3
    for col in numerical_columns:
        z_scores = zscore(df[col])
        df = df[np.abs(z_scores) <= outlier_threshold]
    
    # Convert categorical variables to category datatype
    for col in categorical_columns:
        df[col] = df[col].astype('category')
    
    # Create directory for processed data if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == '__main__':
    clean_data()
