# Azerbaijan E-Commerce Purchase Prediction — Decision Tree

Predicting customer purchase intent using Decision Tree on 100K+ Azerbaijan e-commerce records with 10 behavioral and demographic features.

## What It Does

Takes a customer's age, previous purchases, time on site, membership type, and other shopping metrics — predicts whether they will make a purchase.

## Pipeline

1. **Cleaning** — Removed duplicates (~498), filled missing values with median, dropped irrelevant columns (ID, Date)
2. **EDA** — Class balance chart, age distribution, amount distribution
3. **Encoding** — LabelEncoder for 5 categorical features
4. **Imbalance** — SMOTE (sampling_strategy=0.5) to handle 84%/16% class imbalance
5. **Modeling** — Baseline overfit model → GridSearchCV (1900 combinations, 5-Fold CV)
6. **Tuning** — GridSearchCV with criterion, max_depth, min_samples_split, min_samples_leaf, class_weight
7. **Evaluation** — Classification Report, Confusion Matrix, Cross-Validation, Feature Importance

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 85% |
| F1 Score (CV) | 88% ± 1.2% |
| Overfitting | Fixed (0.20 → 0.04 gap) |
| Best Model | Gini, max_depth=7, min_samples_split=2 |

Overfitting gap reduced from **0.20 → 0.04** after GridSearchCV. Top feature: **Əvvəlki_Alışlar** (previous purchases).

## Quick Start
```bash
pip install -r requirements.txt
jupyter notebook Decision_Tree.ipynb
```

## Live Demo

[azshop-ai.onrender.com](https://azshop-ai.onrender.com)

---

Elton Valiyev — [LinkedIn](https://linkedin.com/in/eltonvaliyev)
