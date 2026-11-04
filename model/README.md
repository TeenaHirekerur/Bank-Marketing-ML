# Bank Marketing Classification

## Project Overview
This project uses Machine Learning classification algorithms to predict whether a customer will subscribe to a bank term deposit.

## Dataset
- Source: UCI Bank Marketing Dataset
- Total Records: 45,211
- Features: 16
- Target: Subscription (Yes/No)

## Machine Learning Models
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

## Model Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Matthews Correlation Coefficient (MCC)

## Application
A Streamlit web application allows users to upload test data and compare the performance of all five trained models.

## Technologies Used
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Joblib.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
