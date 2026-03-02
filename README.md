# 📌 Telco Customer Churn Prediction – End-to-End ML Pipeline

## 🚀 Project Overview

Customer churn is a critical business problem for subscription-based companies.  
Predicting which customers are likely to leave enables proactive retention strategies and reduces revenue loss.

In this project, I built a complete end-to-end Machine Learning pipeline to predict customer churn using the IBM Telco dataset.

This project demonstrates:

- Structured data preprocessing using ColumnTransformer
- Model benchmarking across multiple algorithms
- Hyperparameter tuning with GridSearchCV
- Handling class imbalance
- Feature importance analysis
- Final model selection and persistence

---

## 🎯 Business Objective

Predict whether a customer will churn (Yes/No) based on:

- Demographics
- Service subscriptions
- Contract type
- Billing information
- Account tenure

The goal is to identify high-risk customers and support retention strategies.

---

## 📊 Dataset

Dataset: IBM Telco Customer Churn  
Rows: 7,043  
Features used for modeling: 19  

### Feature Categories:
- Demographic Information
- Account Information
- Services Subscribed
- Billing Details
- Target Variable: Churn

---

## 🛠 Machine Learning Workflow

### 1️⃣ Data Cleaning
- Converted `TotalCharges` to numeric
- Removed missing values
- Encoded target variable (Churn → 0/1)
- Dropped `customerID`

### 2️⃣ Train-Test Split
- 80/20 split
- Stratified sampling to preserve class distribution

### 3️⃣ Feature Engineering
- Numerical features → Median Imputation + Standard Scaling
- Categorical features → Most Frequent Imputation + OneHotEncoding
- Implemented using ColumnTransformer

### 4️⃣ Model Benchmarking

Three models were evaluated:

- Logistic Regression (Baseline)
- Random Forest
- Gradient Boosting

All models were tuned using GridSearchCV with 5-fold cross-validation and optimized for F1-score due to class imbalance.

---

## 📈 Model Performance Comparison

| Model | Accuracy | Recall (Churn) | Precision (Churn) | F1 (Churn) |
|--------|----------|---------------|-------------------|------------|
| Logistic Regression (Tuned) | 0.73 | 0.79 | 0.49 | 0.61 |
| Random Forest (Tuned) | 0.77 | 0.75 | 0.55 | 0.63 |
| Gradient Boosting | 0.79 | 0.53 | 0.63 | 0.58 |

---

## 🏆 Final Model Selection

Random Forest was selected as the final production model because it provides:

- Strong overall balance
- Highest F1-score for churn prediction
- Better precision than Logistic Regression
- Better recall balance than Gradient Boosting

This makes it suitable for real-world churn intervention strategies.

---

## 🔎 Key Business Insights (Feature Importance)

Top drivers of churn include:

- Contract Type (Month-to-Month contracts show higher churn risk)
- Tenure (Short-term customers churn more frequently)
- Monthly Charges
- Total Charges
- Internet Service Type

These insights provide actionable strategies such as promoting long-term contracts and targeting new customers with retention campaigns.

---

## 💾 Production Model

Final model saved as:

models/final_random_forest_churn_model.pkl

To load the model:

```python
import joblib

model = joblib.load("models/final_random_forest_churn_model.pkl")
```

---

## 🏗 Project Structure

```
telco-churn-ml-pipeline/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ └── eda.ipynb
│
├── src/
├── models/
├── README.md
├── requirements.txt
└── .gitignore

```

--- 

## 🔬 Future Improvements

- Deploy model using Streamlit
- Add SHAP for advanced interpretability
- Experiment with XGBoost or LightGBM
- Implement cost-sensitive evaluation metrics

---

## 🏁 Conclusion

This project demonstrates a complete machine learning lifecycle:

- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter tuning
- Model comparison
- Business interpretation
- Production model saving

It reflects real-world ML engineering practices and decision-making processes.

---

## 🚀 Technologies Used

- Python

- Pandas

- NumPy

- Scikit-learn

- Matplotlib

- Seaborn

- Joblib