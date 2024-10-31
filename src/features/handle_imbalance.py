
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import os

def handle_imbalance(input_path=r'C:\Users\Denny\insurance-risk-modeling\data\processed\final_insurance_data.csv', 
                     output_train_path=r'C:\Users\Denny\insurance-risk-modeling\data\processed\train_balanced_data.csv',
                     output_test_path=r'C:\Users\Denny\insurance-risk-modeling\data\processed\test_balanced_data.csv',
                     method='smote', test_size=0.2, random_state=42):
    # Load the dataset
    df = pd.read_csv(input_path)

    # Separate features and target
    X = df.drop(columns='Accident')
    y = df['Accident']

    # Check class distribution
    class_counts = y.value_counts()
    print(f"Class distribution before balancing:\n{class_counts}\n")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Apply balancing techniques based on the specified method
    if method == 'smote':
        smote = SMOTE(random_state=random_state)
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    elif method == 'undersample':
        undersample = RandomUnderSampler(random_state=random_state)
        X_train_bal, y_train_bal = undersample.fit_resample(X_train, y_train)
    elif method == 'smoteenn':
        smote_enn = SMOTEENN(random_state=random_state)
        X_train_bal, y_train_bal = smote_enn.fit_resample(X_train, y_train)
    else:
        raise ValueError("Method must be 'smote', 'undersample', or 'smoteenn'")

    # Check class distribution after balancing
    balanced_class_counts = y_train_bal.value_counts()
    print(f"Class distribution after balancing with {method}:\n{balanced_class_counts}\n")

    # Combine balanced train set for easy saving
    train_balanced_df = pd.concat([X_train_bal, y_train_bal], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)

    # Save the balanced train and original test sets
    train_balanced_df.to_csv(output_train_path, index=False)
    test_df.to_csv(output_test_path, index=False)
    print(f"Balanced training data saved to {output_train_path}")
    print(f"Test data saved to {output_test_path}")

if __name__ == '__main__':
    handle_imbalance()
