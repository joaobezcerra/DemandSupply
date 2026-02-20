# Machine Learning to Predict Transportation Demand

## About the Project
This project uses Machine Learning to predict transportation demand based on historical data. The goal is to optimize vehicle allocation, improving the efficiency of transportation services.

## Dataset Used
The data was obtained from Kaggle's **[Rides Demand and Supply Dataset](https://www.kaggle.com/datasets)**. This dataset contains information about the demand and supply of rides, such as:
- **Active drivers per hour**
- **Active passengers per hour**
- **Completed rides**

## Technologies Used
- **Python**
- **Pandas** (for data manipulation)
- **Scikit-learn** (for predictive modeling)
- **Matplotlib & Seaborn** (for data visualization)

## Data Visualization
The project includes exploratory data analysis and graphical display to understand demand patterns and predictive model performance.

### Run the Prediction Script
```bash
python demandatransporte.py
```

## Validation Method
The selected method is **5-fold cross-validation (K-Fold with shuffle)**.

Why this method?
- The current dataset has no explicit timestamp column.
- Therefore, a temporal split could not be applied safely.
- K-Fold reduces dependency on a single train/test split by rotating validation folds.

## Model Comparison
Two models are evaluated under the same validation protocol:
- **Baseline model** (`DummyRegressor` with mean of `y_train`)
- **Linear Regression model**

For each model, the script reports:
- **Mean ± standard deviation** of **MAE**, **MSE**, and **R²** across the 5 folds.

## How to Interpret the Results
- Lower **MAE** and **MSE** indicate smaller prediction errors.
- Higher **R²** indicates better explanation of variability in completed rides.
- If Linear Regression outperforms the baseline consistently, the features carry useful predictive signal.
- The reported standard deviation indicates stability: lower deviation means more consistent performance across folds.
