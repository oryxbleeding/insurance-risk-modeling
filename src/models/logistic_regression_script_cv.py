# Updating the logistic regression and random forest scripts to include cross-validation functionality
import os
# Updated logistic regression training script with cross-validation
logistic_regression_script_cv = """
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
    print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\\n", classification_report(y_test, y_pred))

    # Save the best model
    model_save_path = 'src/models/logistic_regression_model.joblib'
    joblib.dump(best_model, model_save_path)
    print(f"\\nBest model saved to '{model_save_path}'")

if __name__ == '__main__':
    train_logistic_regression()
"""

# Updated random forest training script with cross-validation
random_forest_script_cv = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_random_forest(input_path='data/processed/train_balanced_data.csv', test_path='data/processed/test_balanced_data.csv'):
    # Load the training and test datasets
    train_df = pd.read_csv(input_path)
    test_df = pd.read_csv(test_path)

    # Separate features and target
    X_train = train_df.drop(columns='Accident')
    y_train = train_df['Accident']
    X_test = test_df.drop(columns='Accident')
    y_test = test_df['Accident']

    # Initialize random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())

    # Train on the full training set and evaluate on the test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("Random Forest Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\\n", classification_report(y_test, y_pred))

if __name__ == '__main__':
    train_random_forest()
"""

# Define paths for saving the updated scripts
os.makedirs("src/models", exist_ok=True)
logistic_regression_path_cv = "src/models/train_logistic_regression.py"
random_forest_path_cv = "src/models/train_random_forest.py"

# Write the updated content to the respective files
with open(logistic_regression_path_cv, "w") as f:
    f.write(logistic_regression_script_cv)

with open(random_forest_path_cv, "w") as f:
    f.write(random_forest_script_cv)

logistic_regression_path_cv, random_forest_path_cv
