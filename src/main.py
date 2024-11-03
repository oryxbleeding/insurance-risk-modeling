import os
from pathlib import Path
from data.clean_data import clean_data
from features.feature_engineering import feature_engineering_and_split
from features.handle_imbalance import handle_imbalance
from models.train_logistic_regression import train_logistic_regression
from models.random_forest_tuning import train_random_forest

def run_pipeline():
    print("Starting Insurance Risk Modeling Pipeline...")
    
    # Step 1: Clean the raw data
    print("\n1. Cleaning data...")
    clean_data(
        input_path='insurance_data.csv',
        output_path='cleaned_insurance_data.csv'
    )
    
    # Step 2: Feature Engineering and Initial Split
    print("\n2. Performing feature engineering...")
    feature_engineering_and_split(
        input_path='data/processed/cleaned_insurance_data.csv',
        train_output_path='data/processed/train_data.csv',
        test_output_path='data/processed/test_data.csv'
    )
    
    # Step 3: Handle Class Imbalance
    print("\n3. Handling class imbalance...")
    handle_imbalance(
        input_path='data/processed/train_data.csv',
        output_train_path='data/processed/train_balanced_data.csv',
        output_test_path='data/processed/test_balanced_data.csv',
        method='smote'
    )
    
    # Step 4: Train Models
    print("\n4. Training Logistic Regression model...")
    train_logistic_regression()
    
    print("\n5. Training Random Forest model...")
    train_random_forest()
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()