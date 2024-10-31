
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def feature_engineering_and_split(input_path='data/processed/cleaned_insurance_data.csv',
                                  train_output_path='data/processed/train_data.csv',
                                  test_output_path='data/processed/test_data.csv',
                                  test_size=0.2, random_state=42):
    # Load the cleaned dataset
    df = pd.read_csv(input_path)
    
    # Feature Engineering: Creating Risk_Age_Group based on Age
    bins = [17, 25, 35, 50, 65, np.inf]
    labels = ['18-25', '26-35', '36-50', '51-65', '65+']
    df['Risk_Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # Interaction Feature: Combining Age and Driving_Experience to create a 'Experience_Age' interaction term
    df['Experience_Age'] = df['Age'] * df['Driving_Experience']

    # Encoding categorical variables
    # One-Hot Encoding for Vehicle_Type and Region
    df = pd.get_dummies(df, columns=['Vehicle_Type', 'Region'], drop_first=True)
    
    # Label Encoding for Risk_Age_Group
    df['Risk_Age_Group'] = df['Risk_Age_Group'].cat.codes
    
    # Feature Scaling: Standardizing numerical columns
    numerical_columns = ['Age', 'Previous_Accidents', 'Annual_Mileage', 'Premium', 'Experience_Age']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Create directory for processed data if it does not exist
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    
    # Save the train and test data
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    
    print(f"Training data saved to {train_output_path}")
    print(f"Test data saved to {test_output_path}")

if __name__ == '__main__':
    feature_engineering_and_split()
