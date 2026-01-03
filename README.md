# 💳 Credit Card Fraud Detection using Tuned XGBoost

A supervised machine learning project focused on detecting **highly imbalanced credit card fraud data** using **XGBoost**, feature selection, and **Bayesian hyperparameter optimization**.

---

## Overview

Fraud detection datasets are notoriously imbalanced, making accuracy a misleading metric.

This project builds a **robust fraud classification pipeline** that:
- Handles severe class imbalance
- Selects informative features
- Tunes model hyperparameters efficiently
- Optimizes for **F1-score**, not raw accuracy

---

## Dataset

- **Source:** Credit card transactions dataset
- **Target variable:** `Class`
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction

---

## Problem Challenges

- Extreme class imbalance
- Risk of overfitting
- High cost of false negatives
- Feature redundancy

---

## Modeling Approach

### 🔹 Baseline Model
- **XGBoost Classifier**
- Adjusted `scale_pos_weight` to handle imbalance
- Evaluated using confusion matrix and classification report

---

### 🔹 Feature Selection
- Applied **SelectKBest (ANOVA F-test)**
- Reduced dimensionality to top-k features
- Improved generalization and training efficiency

---

### 🔹 Hyperparameter Optimization
Used **Bayesian Optimization** to tune:
- `max_depth`
- `learning_rate`
- `n_estimators`
- `scale_pos_weight`
- Number of selected features (`k`)

Optimization objective:
> **Maximize F1-score using cross-validation**

---

## Final Model

- Trained using optimized hyperparameters
- Evaluated on a held-out test set
- Compared against baseline model

Metrics reported:
- Confusion Matrix
- Precision, Recall, F1-score

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Bayesian Optimization (`bayes_opt`)

---

## Key Takeaways

- Class imbalance must be explicitly addressed
- Feature selection improves robustness
- Bayesian Optimization is more efficient than grid search
- F1-score is critical in fraud detection problems

---

## Project Status

Complete — serves as a **reference pipeline** for imbalanced classification problems in finance and security domains.
