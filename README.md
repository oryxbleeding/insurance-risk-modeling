# Insurance Risk Modeling Project

## Overview
This project implements a machine learning pipeline for predicting insurance accident risks. It includes data preprocessing, feature engineering, model training, and evaluation components, specifically designed to handle insurance data with features like age, driving experience, and vehicle type.

## Project Structure
```
insurance-risk-modeling/
├── data/
│   ├── raw/                  # Raw insurance data
│   └── processed/            # Cleaned and processed datasets
├── notebooks/                # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_univariate_analysis.ipynb
│   ├── 03_bivariate_analysis.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_model_interpretation.ipynb
├── src/
│   ├── data/
│   │   └── clean_data.py     # Data cleaning functions
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── handle_imbalance.py
│   ├── models/
│   │   ├── train_logistic_regression.py
│   │   ├── train_random_forest.py
│   │   └── random_forest_tuning.py
│   └── main.py               # Main pipeline script
├── reports/                  # Generated analysis reports
│   └── model_results.md
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```
git clone https://github.com/oryxbleeding/insurance-risk-modeling.git
```
cd insurance-risk-modeling


2. Create and activate a virtual environment (optional but recommended):
``` 	
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install required packages:
```
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline
To run the entire modeling pipeline from data cleaning to model training:
```
python src/main.py
```


### Individual Components
You can also run individual components:

1. Data Cleaning:

``` 
from src.data.clean_data import clean_data
clean_data()
```


2. Feature Engineering:

```  
from src.features.feature_engineering import feature_engineering_and_split
feature_engineering_and_split()
```


3. Model Training:

```  
from src.models.train_logistic_regression import train_logistic_regression
from src.models.train_random_forest import train_random_forest
train_logistic_regression()
train_random_forest()
```


## Model Details

The project implements two main models:
- Logistic Regression with GridSearchCV for hyperparameter tuning
- Random Forest Classifier with cross-validation

Key features used for prediction:
- Age
- Gender
- Driving Experience
- Vehicle Type
- Previous Accidents
- Region
- Annual Mileage
- Premium

## Results

Model performance and business implications can be found in `reports/model_results.md`. Key findings include:
- Annual Mileage (21.2%) and Premium (21.1%) are the most influential features
- Experience Age (17.1%) significantly affects accident risk
- Age (14.5%) and Driving Experience (11.5%) show moderate importance

## Contact

daniel.do@hotmail.de
Project Link: https://github.com/oryxbleeding/insurance-risk-modeling