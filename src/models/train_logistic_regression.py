
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def train_logistic_regression(input_path='data/processed/train_balanced_data.csv', test_path='data/processed/test_balanced_data.csv'):
    # Load the training and test datasets
    train_df = pd.read_csv(input_path)
    test_df = pd.read_csv(test_path)

    # Separate features and target
    X_train = train_df.drop(columns='Accident')
    y_train = train_df['Accident']
    X_test = test_df.drop(columns='Accident')
    y_test = test_df['Accident']

    # Define hyperparameter grid
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }

    # Initialize model
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Print best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Print evaluation metrics
    print("Logistic Regression Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the best model
    model_save_path = 'src/models/logistic_regression_model.joblib'
    joblib.dump(best_model, model_save_path)
    print(f"\nBest model saved to '{model_save_path}'")

if __name__ == '__main__':
    train_logistic_regression()
