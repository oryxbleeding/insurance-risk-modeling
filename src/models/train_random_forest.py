import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def train_random_forest(input_path='data/processed/train_balanced_data.csv', 
                        test_path='data/processed/test_balanced_data.csv', 
                        model_output_path="../src/models/random_forest_model.pkl"):
    # Load the training and test datasets
    train_df = pd.read_csv(input_path)
    test_df = pd.read_csv(test_path)

    # Separate features and target
    X_train = train_df.drop(columns='Accident')
    y_train = train_df['Accident']
    X_test = test_df.drop(columns='Accident')
    y_test = test_df['Accident']

    # Define hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the random forest model
    model = RandomForestClassifier(random_state=42)

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Print evaluation metrics
    print("Random Forest Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the best model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(best_model, model_output_path)
    print(f"Best model saved to '{model_output_path}'")

if __name__ == '__main__':
    train_random_forest()