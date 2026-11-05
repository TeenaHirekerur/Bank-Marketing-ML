# Bank Marketing Classification

## a. Problem Statement

The objective of this project is to predict whether a customer will subscribe to a bank term deposit using machine learning classification algorithms.

Five classification models are implemented and evaluated on the same dataset.

## b. Dataset Description

- **Dataset:** UCI Bank Marketing Dataset (`bank-full.csv`)
- **Total Instances:** 45,211
- **Total Features:** 16
- **Target Variable:** `y`
- **Target Classes:** Yes / No
- **Problem Type:** Binary Classification

## c. GitHub Repository Link

https://github.com/TeenaHirekerur/Bank-Marketing-ML

## d. Models Used

The following five classification models are implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7548 | 0.7722 | 0.2662 | 0.6238 | 0.3732 | 0.2853 |
| Decision Tree | 0.8401 | 0.6036 | 0.3083 | 0.2949 | 0.3014 | 0.2113 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.7578 | 0.7718 | 0.2684 | 0.6200 | 0.3746 | 0.2867 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved a recall of 0.6238 and AUC of 0.7722. |
| Decision Tree | Achieved an accuracy of 0.8401, but comparatively lower recall (0.2949) and AUC (0.6036). |
| KNN | Achieved the highest accuracy (0.8874) and precision (0.5529), but the lowest recall (0.1975). |
| Naive Bayes | Achieved the highest F1-score (0.4123) and MCC (0.3271). |
| Support Vector Machine | Achieved a recall of 0.6200 and AUC of 0.7718, showing performance similar to Logistic Regression. |

### Overall Winner for the Dataset

**Naive Bayes** is the overall winner based on its highest F1-score (0.4123) and MCC (0.3271), indicating the best balance among the evaluated models.
