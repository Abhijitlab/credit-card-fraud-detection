# Credit Card Fraud Detection using Machine Learning

End-to-end machine learning project to detect fraudulent credit card transactions on a highly imbalanced real-world dataset.

## Project Overview

This project builds a complete fraud detection pipeline using the popular Credit Card Fraud Detection dataset (284,807 transactions with only 0.172% fraud cases). The focus is on handling extreme class imbalance and evaluating models with metrics that matter in real-world fraud systems.

### Key Highlights
- Handled severe class imbalance using **SMOTE**
- Compared **Logistic Regression**, **Random Forest**, and **XGBoost**
- Performed **threshold tuning** with business cost analysis
- Used **SHAP** for model explainability
- Evaluated using Precision, Recall, F1-Score, ROC-AUC, and PR-AUC

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud Rate**: 0.172% (492 fraudulent transactions)
- **Features**: Time, Amount, and 28 PCA-transformed features (V1–V28)

## Approach

1. Exploratory Data Analysis (class distribution, amount patterns)
2. Data Preprocessing (scaling Time & Amount)
3. Stratified Train-Test Split
4. Handling Imbalance with SMOTE (applied only on training data)
5. Model Training & Comparison
6. Threshold Tuning + Business Cost Analysis
7. Model Explainability using SHAP
8. Model Persistence

## Models Used

| Model                  | Purpose                          |
|------------------------|----------------------------------|
| Logistic Regression    | Interpretable baseline           |
| Random Forest          | Strong ensemble model            |
| XGBoost                | High-performance gradient boosting |

## Evaluation Metrics

Because of extreme class imbalance, accuracy is **not** used as the main metric. Instead, the project focuses on:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC (Average Precision)

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- SHAP
- Matplotlib, Seaborn
- joblib

## How to Run

1. Open the notebook in Google Colab or Jupyter
2. Run all cells in order
3. The trained model will be saved as `fraud_detection_model.pkl`

## Results

The project successfully demonstrates:
- Strong ranking performance (high ROC-AUC & PR-AUC)
- Practical threshold selection based on business cost
- Feature importance understanding using SHAP

## Future Improvements

- Deploy as a Streamlit / FastAPI application
- Add real-time prediction API
- Experiment with anomaly detection techniques (Isolation Forest, Autoencoders)
- Include cost-sensitive learning

## Author

Your Name  
Aspiring Data Scientist / Machine Learning Engineer

---

**Note**: This project was built as a practical demonstration of handling real-world imbalanced classification problems in the financial domain.
