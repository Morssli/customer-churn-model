# Customer Churn Prediction Project

## Overview
This project predicts customer churn using a logistic regression model on the Telco Customer Churn dataset. The workflow includes data preprocessing, EDA, feature engineering, model training, and evaluation.

## Results Summary
| Metric      | Value  |
|------------|--------|
| Accuracy   | 0.805  |
| Precision  | 0.655  |
| Recall     | 0.559  |
| F1 Score   | 0.603  |
| AUC-ROC    | 0.842  |

- The model achieves strong performance, especially in AUC-ROC.
- Top features driving churn are visualized in the script.

## Visualizations
- **Confusion Matrix**: Shows model prediction breakdown for churn vs. no churn.
- **ROC Curve**: Illustrates model discrimination ability.
- **Top Features Bar Chart**: Highlights the most influential features for churn.

## How to Run
1. Install dependencies from `requirements.txt`.
2. Run `churn_model.py` to train the model and view results.

## Project Structure
```
├── README.md
├── churn_model.py
├── requirements.txt
├── .gitignore
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```


