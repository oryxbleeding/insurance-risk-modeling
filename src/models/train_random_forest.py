
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def train_random_forest(input_path=r"C:\Users\Denny\insurance-risk-modeling\data\processed\train_balanced_data.csv", test_path=r'C:\Users\Denny\insurance-risk-modeling\data\processed\test_balanced_data.csv'):
    # Load the training and test datasets
    train_df = pd.read_csv(input_path)
    test_df = pd.read_csv(test_path)

    # Separate features and target
    X_train = train_df.drop(columns='Accident')
    y_train = train_df['Accident']
    X_test = test_df.drop(columns='Accident')
    y_test = test_df['Accident']

    # Initialize and train the random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("Random Forest Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_random_forest()
