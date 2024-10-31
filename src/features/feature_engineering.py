
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def feature_engineering(input_path='data/processed/cleaned_insurance_data.csv', output_path='data/processed/final_insurance_data.csv'):
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
    
    # Create directory for processed data if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the final dataset with engineered features
    df.to_csv(output_path, index=False)
    print(f"Final dataset with engineered features saved to {output_path}")

if __name__ == '__main__':
    feature_engineering()
