import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def train_logistic_regression(input_path=r'C:\Users\Denny\insurance-risk-modeling\data\processed\train_balanced_data.csv', test_path=r'C:\Users\Denny\insurance-risk-modeling\data\processed\test_balanced_data.csv'):
    # Lade Trainings- und Testdaten
    train_df = pd.read_csv(input_path)
    test_df = pd.read_csv(test_path)

    # Features und Zielvariablen trennen
    X_train = train_df.drop(columns='Accident')
    y_train = train_df['Accident']
    X_test = test_df.drop(columns='Accident')
    y_test = test_df['Accident']

    # Definiere Hyperparameter-Grid für das Tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }

    # Initialisiere das Modell
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Führe Grid Search mit Cross-Validation durch
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Beste Parameter und Score ausgeben
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Das beste Modell auf den Testdaten evaluieren
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metriken ausgeben
    print("Logistic Regression Model Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the best model
    ##os.makedirs('models', exist_ok=True)
    model_save_path = 'src/models/logistic_regression_model.joblib'
    joblib.dump(best_model, model_save_path)
    print(f"\nBest model saved to '{model_save_path}'")

if __name__ == '__main__':
    train_logistic_regression()
