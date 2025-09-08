# GTC-ML-Project-2---Diabetes-Prediction-Model
# 🩺 Diabetes Prediction using Machine Learning

## 📌 Project Overview
This project aims to build a **machine learning model** that predicts whether a patient is **Diabetic** or **Non-Diabetic** based on medical features such as glucose level, BMI, age, and more.  
We implemented multiple models (Logistic Regression, SVM, Random Forest), performed hyperparameter tuning, and created a **prediction engine** that can classify new patient data.

---

## 🚀 Phases of the Project

### **Phase 1: Data Exploration (EDA)**
- Explored the dataset to understand patterns and distributions.
- Answered key questions:
  - How many patients have diabetes vs. those who don’t?
  - What’s the relationship between **Glucose** and the outcome?
  - Does **BMI** play a significant role?
- Used graphs (scatter plots, histograms, correlation heatmaps) to uncover insights.

---

### **Phase 2: Data Preparation**
- Handled missing values.
- Standardized features using **StandardScaler** to ensure all variables are on the same scale.
- Split data into **training (80%)** and **testing (20%)** sets.

---

### **Phase 3: Model Building**
Implemented multiple models:
- ✅ **Logistic Regression**
- ✅ **Support Vector Machine (SVM)**
- ✅ **Random Forest Classifier** (best performer after tuning)

Performed **GridSearchCV** for hyperparameter tuning:
- Random Forest Best Params:  
  `{'class_weight': 'balanced', 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 1, 'n_estimators': 500}`

---

### **Phase 4: Prediction Engine**
Built a function that:
- Takes new patient data as input.
- Returns **"Diabetic"** or **"Non-Diabetic"** instantly.
- Example:
```python

new_patient = [1, 85, 66, 29, 0, 26.6, 0.351, 31]
Non-Diabetic
new_patient = [6,	148,	72,	35,	0,	33.6,	0.627,	50]
Diabetic

Random Forest (Best Model):

Accuracy: ~75%

Precision (Class 0): 0.79

Recall (Class 1): 0.57

F1-Score (Class 1): 0.61

👉 Performance improved after tuning hyperparameters.

🛠 Tech Stack

Python 🐍

Pandas, NumPy (Data Processing)

Matplotlib, Seaborn (Visualization)

Scikit-learn (ML Models & Evaluation)

💡 Future Improvements

Collect more patient data for better generalization.

Use advanced models like XGBoost or LightGBM.

Deploy the prediction engine using Streamlit for a web app.

Add probability outputs (e.g., "78% chance Diabetic").

Author: Nada Maher Mohamed Elhady

