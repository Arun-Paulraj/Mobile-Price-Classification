# 📱 Mobile Price Classification Using Machine Learning

---

# A. Problem Statement

Mobile phone pricing is influenced predominantly by technical specifications. Informed decisions can be made by manufacturers, retailers and consumers when the price category of a device and be identified accurately.

The objective of this project is to predict the price category of a mobile phone based on its hardware and technical specifications. Twenty features such as battery power, RAM, internal memory, processor cores, camera resolution, display characteristics, and connectivity options are used as input variables.

To accomplish this task, five machine learning classification algorithms have been implemented and compared:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)

Each classifier is evaluated using the following performance metrics:

- Accuracy
- AUC (Area Under Curve)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The trained models are used to classify mobile devices into one of the following categories:

| Label | Category |
|---------|---------|
| 0 | Low Cost |
| 1 | Medium Cost |
| 2 | High Cost |
| 3 | Very High Cost |

---

# B. Dataset Description

The Mobile Price Classification dataset contains technical specifications of mobile devices along with their corresponding price category.

## Dataset Summary

| Attribute | Value |
|------------|------------|
| Number of Records | 2000 |
| Number of Features | 20 |
| Target Variable | price_range |
| Number of Classes | 4 |

## Features Used

- battery_power
- blue
- clock_speed
- dual_sim
- fc
- four_g
- int_memory
- m_dep
- mobile_wt
- n_cores
- pc
- px_height
- px_width
- ram
- sc_h
- sc_w
- talk_time
- three_g
- touch_screen
- wifi

## Target Classes

| Class Label | Price Category |
|-------------|----------------|
| 0 | Low Cost |
| 1 | Medium Cost |
| 2 | High Cost |
| 3 | Very High Cost |

---

# C. GitHub Repository Link

**Repository URL**

```text
https://github.com/Arun-Paulraj/Mobile-Price-Classification
```

---

# D. Models Used

## Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|----------|----------|----------|----------|----------|
| Logistic Regression | 0.9575 | 0.9979 | 0.9579 | 0.9575 | 0.9572 | 0.9437 |
| Support Vector Machine | 0.8875 | 0.9815 | 0.8868 | 0.8875 | 0.8868 | 0.8502 |
| Decision Tree | 0.8200 | 0.8800 | 0.8207 | 0.8200 | 0.8202 | 0.7601 |
| Naive Bayes | 0.7525 | 0.9207 | 0.7496 | 0.7525 | 0.7507 | 0.6702 |
| K-Nearest Neighbors (KNN) | 0.6350 | 0.8196 | 0.6340 | 0.6350 | 0.6336 | 0.5138 |

---

# Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| Logistic Regression | Achieved the highest performance across all evaluation metrics. It produced the best Accuracy, AUC, F1 Score, and MCC values. |
| Decision Tree | Delivered good classification performance but was less accurate than Logistic Regression and SVM. |
| K-Nearest Neighbors (KNN) | Initially showed poor performance and required hyperparameter tuning. n_neighbors was initally set to 5. Since the results were very poor we have upgraded the value to 30. Even after tuning, it remained the weakest classifier on this dataset. |
| Naive Bayes | Produced moderate performance with reasonable Accuracy and AUC values. The scores couldn't beat Logistic regression due to its assumption that all input features are independent. |
| Support Vector Machine (SVM) | Achieved the second-best performance and demonstrated strong generalization capability with excellent AUC values. |

---

# Overall Winner for the Dataset

 **Logistic Regression**

Logistic Regression emerged as the best-performing classifier for the Mobile Price Classification dataset.

### Winning Performance

| Metric | Score |
|----------|----------|
| Accuracy | 95.75% |
| AUC | 99.79% |
| Precision | 95.79% |
| Recall | 95.75% |
| F1 Score | 95.72% |
| MCC | 94.37% |

The results indicate that the dataset is highly suitable for linear classification methods, with Logistic Regression significantly outperforming the remaining classifiers.

---

# Streamlit Application Features

The Streamlit application provides:

- Dataset Overview
- Dataset Statistics
- Dataset Preview
- Price Category Distribution
- Machine Learning Classifier Performance Comparison
- Dynamic Performance Metric Visualization
- Test Dataset Evaluation
- Confusion Matrix
- Classification Report
- Mobile Price Prediction
- Prediction Confidence Score

---

# Repository Structure

```text
Mobile-Price-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_training.ipynb
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── metrics.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree_classifier.pkl
│   ├── knn_classifier.pkl
│   ├── naive_bayes_classifier.pkl
│   ├── svm_classifier.pkl
│   └── scaler.pkl
```

---

## Test Dataset for Streamlit Validation

The file **test_data.csv** contains both the feature columns and the target variable (**price_range**) and is intended for use with the **Test Dataset Evaluation** page of the Streamlit application.

This dataset can be uploaded through the Streamlit interface to evaluate the performance of the trained machine learning models. The application uses this file to generate evaluation metrics such as Accuracy, AUC, Precision, Recall, F1 Score, MCC, Confusion Matrix, and Classification Report without retraining the model.

# Conclusion

Five machine learning classifiers were implemented and evaluated for mobile price prediction. Among all models, Logistic Regression achieved the best overall performance and proved to be the most suitable classifier for this dataset. The developed Streamlit application enables users to compare classifier performance, evaluate uploaded test datasets, and predict the price category of mobile devices based on their specifications. 
