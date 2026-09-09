# 💳 Credit Card Fraud Detection

> A Machine Learning project for detecting fraudulent credit card transactions using different classification and sampling techniques.

---

## 📌 Project Overview

Credit card fraud is a major problem in online financial transactions. Since fraudulent transactions are much fewer than normal transactions, the dataset is highly imbalanced.

In this project, Machine Learning techniques are used to identify fraudulent transactions and compare different approaches for handling the imbalanced dataset.

The project explores **Logistic Regression, K-Nearest Neighbors (KNN), Random Under Sampling, and SMOTE Oversampling**.

---

## 🎯 Objectives

- Detect fraudulent credit card transactions.
- Understand and handle class imbalance.
- Compare different Machine Learning approaches.
- Evaluate models using suitable performance metrics.
- Identify a suitable model for fraud detection.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Programming Language |
| 🐼 Pandas | Data Processing |
| 🔢 NumPy | Numerical Operations |
| 📊 Matplotlib | Data Visualization |
| 📈 Seaborn | Data Visualization |
| 🤖 Scikit-learn | Machine Learning |
| ⚖️ Imbalanced-learn | Sampling Techniques |
| 📓 Jupyter Notebook | Development Environment |

---

## 🧠 Machine Learning Models

The following models were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)

### Sampling Techniques

To handle the highly imbalanced dataset:

- Random Under Sampling
- SMOTE (Synthetic Minority Oversampling Technique)

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Check Missing Values
   ↓
Remove Duplicate Records
   ↓
Analyze Class Imbalance
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Sampling Techniques
   ├── Random Under Sampling
   └── SMOTE Oversampling
   ↓
Machine Learning Models
   ├── Logistic Regression
   └── KNN
   ↓
Model Evaluation
   ↓
Model Comparison