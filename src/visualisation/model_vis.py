import joblib
import pandas as pd
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from IPython.display import display

# Lade die Modelle
logistic_model = joblib.load("src/models/logistic_regression_model.joblib")
random_forest_model = joblib.load("src/models/random_forest_model.pkl")

# Zeige die Hyperparameter in DataFrames für eine klare Darstellung
logistic_params = pd.DataFrame(logistic_model.get_params(), index=[0])
rf_params = pd.DataFrame(random_forest_model.get_params(), index=[0])

# Ausgabe der Hyperparameter in DataFrames
print("Logistic Regression Parameters:")
print(logistic_params.T)

print("\nRandom Forest Parameters:")
print(rf_params.T)

# Print number of trees in the model
print(f"Number of trees in the model: {len(random_forest_model.estimators_)}")

# Print feature count from model
n_features = random_forest_model.n_features_in_
print(f"Number of features in model: {n_features}")

# Make sure feature names match the model's feature count
feature_names = ['Age', 'Annual_Mileage', 'Premium', 'Experience_Age', 'Driving_Experience',
                 'Previous_Accidents', 'Risk_Age_Group', 'Region_1', 'Vehicle_Type_1', 
                 'Vehicle_Type_2', 'Vehicle_Type_3', 'Vehicle_Type_4']
print(f"Number of feature names provided: {len(feature_names)}")

# Only plot if feature counts match
if len(feature_names) == n_features:
    plt.figure(figsize=(20, 10))
    plot_tree(random_forest_model.estimators_[0], feature_names=feature_names, filled=True)
    plt.title("Visualization of the First Tree in Random Forest Model")
    plt.show()
else:
    print(f"Feature count mismatch! Model expects {n_features} features but {len(feature_names)} were provided.")
