# Credit Card Fraud Detection

## Overview

This project implements a machine learning-based system for detecting credit card fraud using a highly imbalanced dataset. The goal is to classify transactions as fraudulent or legitimate with high accuracy, focusing on minimizing false negatives (missed frauds) while maintaining low false positives.

The project uses Python and scikit-learn to build and evaluate models, including ensemble methods like Voting Classifier combining Logistic Regression, Random Forest, and Neural Networks. Data preprocessing includes scaling and sampling techniques to handle class imbalance.

## Handling Imbalanced Dataset

The dataset is severely imbalanced, with only ~0.16% of transactions being fraudulent. To address this, we implemented several sampling strategies:

- **Oversampling**: Using SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples for the minority class (fraud).
- **Undersampling**: Using NearMiss and RandomUnderSampler to reduce the majority class (legitimate transactions).
- **Combination**: Applying oversampling or undersampling before training to balance the classes.

These techniques help improve model performance on the minority class without overfitting.

## Files Structure

```
credit_fraud_train.py          # Main training script with model pipeline
credit_fraud_utils_data.py     # Data loading and preprocessing utilities
credit_fraud_utils_eval.py     # Evaluation and results saving functions
EDA.ipynb                      # Exploratory Data Analysis notebook
test_script.py                 # Script to test the trained model on test data
README.md                      # This file
__pycache__/                   # Python cache files
Model/                         # Directory for saved models
Training_Results/              # Directory for training results
    results.json               # JSON file with training and validation metrics
```

## How to Run the Code

### Prerequisites
- Python 3.x
- Required libraries: pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### Training the Model
Run the training script with default parameters:
```bash
python credit_fraud_train.py
```

For custom parameters (e.g., using SMOTE oversampling):
```bash
python credit_fraud_train.py --OverSampling SMOTE --OverSamplingFactor 10 --SaveModel True
```

Key arguments:
- `--dataset`: Path to dataset directory (default: '../data/split')
- `--OverSampling`: 'SMOTE' or 'RandomOverSampling'
- `--UnderSampling`: 'NearMiss' or 'RandomUnderSampling'
- `--SaveModel`: Set to True to save the trained model
- `--SaveResults`: Set to True to save evaluation results

### Testing the Model
After training and saving the model, run the test script:
```bash
python test_script.py
```

Ensure the test data is at '../data/split/test.csv' and the model is at 'Model/VotingClassifier.pkl'.

### Exploratory Data Analysis
Open and run the Jupyter notebook:
```bash
jupyter notebook EDA.ipynb
```

## Final Best Results

Based on validation results, the best performing model used SMOTE oversampling with a Voting Classifier (Logistic Regression + Random Forest). Key metrics:

- **Precision (Fraud)**: 0.87
- **Recall (Fraud)**: 0.82
- **F1-Score (Fraud)**: 0.85