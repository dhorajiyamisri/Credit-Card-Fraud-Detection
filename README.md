<div align="center">

# 💳 Credit Card Fraud Detection

### 🚨 An End-to-End Machine Learning System for Detecting Suspicious Credit Card Transactions

<p>
  <b>Imbalanced Classification • SMOTE • Random Forest • XGBoost • Hyperparameter Tuning • Threshold Optimization</b>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Model-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Imbalanced--Learn-SMOTE-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
</p>

<p>
  <a href="https://github.com/dhorajiyamisri/Credit-Card-Fraud-Detection">
    <img src="https://img.shields.io/badge/🔗%20GitHub-Repository-black?style=for-the-badge&logo=github" />
  </a>
</p>

</div>

---

## 📌 Overview

**Credit Card Fraud Detection** is a machine learning classification project designed to identify potentially fraudulent credit card transactions.

The project focuses on one of the most important challenges in financial machine learning: **extreme class imbalance**.

In the original dataset:

* **284,807 transactions**
* **31 columns**
* Legitimate transactions (`Class = 0`) dominate the dataset.
* Fraudulent transactions (`Class = 1`) represent only about **0.17%** of the transactions.

Because fraud is extremely rare, simply maximizing accuracy is not enough. The project therefore focuses heavily on:

* Precision
* Recall
* F1 Score
* Precision-Recall AUC (PR-AUC)
* Threshold optimization

The final workflow compares **Logistic Regression, Random Forest and XGBoost**, followed by hyperparameter tuning and decision-threshold optimization.

---

# 🎯 Business Problem

Credit card fraud is a highly imbalanced classification problem.

A financial institution may process thousands or millions of legitimate transactions while only a very small number are fraudulent.

The objective is:

> **Given transaction-level features, classify whether a transaction is legitimate or potentially fraudulent.**

### Why is this challenging?

A model could classify almost every transaction as legitimate and still achieve extremely high accuracy because fraudulent transactions are rare.

For example:

```text
Legitimate Transactions  → 99.8%+
Fraudulent Transactions  → ~0.17%
```

Therefore:

> **High Accuracy ≠ Good Fraud Detection**

A fraud detection system must successfully identify fraudulent transactions while controlling the number of legitimate transactions incorrectly flagged as fraud.

---

# 🧠 Machine Learning Objective

This project solves a **binary classification problem**.

### Target Variable

```text
Class
```

| Class | Meaning                |
| ----: | ---------------------- |
|   `0` | Legitimate Transaction |
|   `1` | Fraudulent Transaction |

The model ultimately produces a probability representing how likely a transaction is to belong to the fraud class.

---

# 📊 Dataset

The notebook loads the dataset using:

```python
df = pd.read_csv("creditcard.csv")
```

### Dataset dimensions

```text
284,807 rows
31 columns
```

The target distribution is approximately:

```text
Class 0 → 99.8273%
Class 1 → 0.1727%
```

This makes the dataset **highly imbalanced**.

---

# 🔎 Exploratory Data Analysis

The project performs exploratory analysis before model development.

### 1. Class Distribution

A count plot with logarithmic y-axis is used to visualize the extreme difference between legitimate and fraudulent transactions.

```python
sns.countplot(x='Class', data=sample)
plt.yscale('log')
```

### 2. Transaction Amount Distribution

The project also compares transaction amount distributions between:

* Legitimate transactions
* Fraudulent transactions

```python
sns.histplot(
    sample[sample['Class']==0]['Amount'],
    label='Legit',
    kde=True
)

sns.histplot(
    sample[sample['Class']==1]['Amount'],
    label='Fraud',
    kde=True
)
```

---

# 🧹 Data Preprocessing & Feature Engineering

The preprocessing pipeline contains several important steps.

## 1. Log Transformation of Amount

Transaction amounts can have a skewed distribution.

The project creates:

```python
df['Amount_log'] = np.log1p(df['Amount'])
```

This transforms the original amount feature into a more manageable scale.

---

## 2. Extract Transaction Hour

The original `Time` feature is converted into an hourly representation:

```python
df['Hour'] = (df['Time'] % 86400) // 3600
```

This allows the model to capture potential time-of-day patterns.

---

## 3. Remove Original Features

After feature engineering:

```python
df = df.drop(['Time','Amount'], axis=1)
```

The original `Time` and `Amount` columns are removed because their transformed representations are now used.

---

# ⚖️ Feature Scaling

The engineered features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

df[['Amount_log','Hour']] = scaler.fit_transform(
    df[['Amount_log','Hour']]
)
```

After scaling, the transformed features have approximately:

```text
Mean ≈ 0
Standard Deviation ≈ 1
```

---

# ✂️ Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

### Why `stratify=y`?

Because fraud is extremely rare, random splitting without stratification could create an undesirable class distribution.

Stratification preserves approximately the same fraud ratio in both training and testing data.

---

# ⚠️ Handling Extreme Class Imbalance

The training set contains:

```text
Legitimate → 227,451
Fraud      → 394
```

This is a severe imbalance.

The project experiments with two major resampling strategies.

---

## 🔵 Strategy 1 — SMOTE

**SMOTE = Synthetic Minority Over-sampling Technique**

SMOTE generates synthetic minority-class samples instead of simply duplicating existing fraud transactions.

The project uses:

```python
SMOTE(
    random_state=42,
    sampling_strategy=0.1
)
```

Result:

```text
Majority Class → 227,451
Fraud Class    → 22,745
```

The minority class is increased to approximately **10% of the majority class**.

---

## 🟠 Strategy 2 — Random Undersampling

Random undersampling reduces the number of majority-class examples.

The project uses:

```python
RandomUnderSampler(
    random_state=42,
    sampling_strategy=0.1
)
```

Result:

```text
Majority Class → 3,940
Fraud Class    → 394
```

This creates a much smaller but more balanced training dataset.

---

# 🤖 Machine Learning Models

The project compares multiple machine learning approaches.

```text
Logistic Regression
        ↓
Random Forest
        ↓
XGBoost Baseline
        ↓
XGBoost Hyperparameter Tuning
        ↓
Threshold Optimization
```

---

# 1️⃣ Logistic Regression

Three Logistic Regression variants are evaluated:

### Variant A — Original Imbalanced Data

Uses:

```python
class_weight='balanced'
```

### Variant B — SMOTE

The model is trained on the SMOTE-resampled training data.

### Variant C — Random Undersampling

The model is trained on the undersampled dataset.

### Verified Results

| Strategy                   | Precision | Recall |     F1 | PR-AUC |
| -------------------------- | --------: | -----: | -----: | -----: |
| Original + Balanced Weight |    0.0555 | 0.9082 | 0.1046 | 0.7128 |
| SMOTE                      |    0.3686 | 0.8878 | 0.5210 | 0.7418 |
| Undersampling              |    0.3372 | 0.8878 | 0.4888 | 0.7449 |

### Observation

The original balanced Logistic Regression achieves high fraud recall but very low precision.

SMOTE substantially improves precision and F1 Score.

---

# 2️⃣ Random Forest

Random Forest is trained using the SMOTE-resampled training data.

Configuration:

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### Verified Test Results

```text
Precision → 0.8367
Recall    → 0.8367
F1 Score  → 0.8367
PR-AUC    → 0.8536
```

This provides a strong improvement over the Logistic Regression variants.

The project also extracts the **Top 15 Feature Importances** from the Random Forest model.

---

# 3️⃣ XGBoost Baseline

The project then introduces **XGBoost**, a powerful gradient boosting algorithm.

Because of the severe class imbalance, the project calculates:

```python
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
```

and uses:

```python
scale_pos_weight=ratio
```

### Baseline Configuration

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    scale_pos_weight=ratio,
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)
```

### Verified Results

```text
Precision → 0.65625
Recall    → 0.85714
F1 Score  → 0.74336
PR-AUC    → 0.85793
```

---

# 4️⃣ XGBoost Hyperparameter Tuning

Instead of relying only on default parameters, the project performs **RandomizedSearchCV**.

### Parameters searched

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
```

The search uses:

```python
RandomizedSearchCV(
    n_iter=15,
    cv=3,
    scoring='average_precision',
    random_state=42,
    n_jobs=-1
)
```

### Why `average_precision`?

For highly imbalanced classification, PR-AUC / Average Precision is more informative than relying only on accuracy.

---

## 🏆 Best Parameters

The notebook identifies:

```text
subsample         = 0.6
n_estimators      = 300
max_depth         = 3
learning_rate     = 0.1
colsample_bytree   = 0.8
```

### Best Cross-Validation PR-AUC

```text
0.842884
```

### Tuned XGBoost Test PR-AUC

```text
0.854052
```

---

# 🎚️ Threshold Optimization

A major strength of this project is that it does **not blindly use the default classification threshold of 0.5**.

Instead, the project analyzes how changing the threshold affects:

```text
Precision
Recall
F1 Score
```

The model produces fraud probabilities:

```python
y_scores = best_xgb.predict_proba(X_test)[:,1]
```

and evaluates different thresholds using the Precision-Recall curve.

---

## 🎯 F1-Optimal Threshold

The notebook calculates the threshold that maximizes F1 Score.

```text
Best Threshold ≈ 0.9747
```

---

## 🚨 Recall ≥ 90% Threshold

The project also searches for a threshold capable of achieving at least 90% recall.

```text
Threshold ≈ 0.1114
```

This demonstrates an important real-world concept:

> **The best threshold depends on the business cost of false positives and false negatives.**

---

# 📈 Evaluation Metrics

Because this is an extremely imbalanced fraud-detection problem, the project focuses on the following metrics.

## Precision

Precision answers:

> Of all transactions predicted as fraud, how many were actually fraud?

```text
Precision = TP / (TP + FP)
```

High precision means fewer legitimate transactions are incorrectly flagged.

---

## Recall

Recall answers:

> Of all actual fraudulent transactions, how many did the model successfully detect?

```text
Recall = TP / (TP + FN)
```

In fraud detection, recall is particularly important because missing a fraudulent transaction can result in financial loss.

---

## F1 Score

F1 Score balances precision and recall.

```text
F1 = 2 × (Precision × Recall)
     -------------------------
       Precision + Recall
```

---

## PR-AUC

The project uses **Precision-Recall AUC / Average Precision** as an important model-selection metric.

This is particularly useful when the positive class is extremely rare.

---

# 📊 Model Performance

### Verified individual model results from the notebook

| Model / Strategy                    |  Precision |     Recall |   F1 Score |     PR-AUC |
| ----------------------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression — Original      |     0.0555 |     0.9082 |     0.1046 |     0.7128 |
| Logistic Regression — SMOTE         |     0.3686 |     0.8878 |     0.5210 |     0.7418 |
| Logistic Regression — Undersampling |     0.3372 |     0.8878 |     0.4888 |     0.7449 |
| Random Forest + SMOTE               | **0.8367** | **0.8367** | **0.8367** | **0.8536** |
| XGBoost Baseline                    |     0.6563 |     0.8571 |     0.7434 | **0.8579** |
| XGBoost Tuned                       |          — |          — |          — | **0.8541** |

> **Note:** The notebook's final comparison section contains approximate (`~`) metric values for some threshold/model variants. The table above intentionally uses only the exact values directly printed by the executed model cells.

---

# 📉 Precision-Recall Analysis

The project generates Precision-Recall curves for:

* Logistic Regression
* Random Forest
* XGBoost Baseline
* Tuned XGBoost

This helps visualize the trade-off between:

```text
Precision ↔ Recall
```

at different classification thresholds.

For fraud detection, this trade-off is critical because increasing fraud detection sensitivity can also increase the number of legitimate transactions flagged for investigation.

---

# 💰 Business Simulation & Cost-Benefit Analysis

The project also introduces a business-oriented fraud detection simulation.

### Assumptions

```text
Average fraudulent transaction value = ₹4,500

Investigation cost per flagged transaction = ₹150
```

### Business formulas

```text
Money Saved
= TP × ₹4,500
```

```text
Investigation Cost
= (TP + FP) × ₹150
```

```text
Money Lost
= FN × ₹4,500
```

```text
Net Benefit
= Money Saved − Investigation Cost
```

This connects the machine learning model to a real-world financial decision.

Instead of asking only:

> "Which model has the highest score?"

the project asks:

> **"Which decision threshold provides the best business outcome?"**

---

# 🔬 Threshold Sensitivity Analysis

The project evaluates different thresholds:

```text
0.1
0.2
0.3
0.5
F1-optimal threshold
```

For each threshold, the business impact can be evaluated using:

* True Positives
* False Positives
* False Negatives
* Money Saved
* Investigation Cost
* Money Lost
* Net Benefit

This provides a bridge between **machine learning metrics and business decision-making**.

---

# 🔄 End-to-End ML Pipeline

```text
┌──────────────────────────────┐
│      Credit Card Data        │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     Data Loading & EDA       │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│   Feature Engineering        │
│   • Amount_log               │
│   • Hour                     │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       Feature Scaling        │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      Stratified Split        │
│         80% / 20%            │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│   Imbalance Handling         │
│   • SMOTE                    │
│   • Undersampling            │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      Model Training          │
│   • Logistic Regression      │
│   • Random Forest             │
│   • XGBoost                  │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│    Hyperparameter Tuning     │
│       RandomizedSearchCV     │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│    Threshold Optimization    │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│ Business Cost-Benefit        │
│       Analysis               │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│    Final ML Pipeline         │
│   + Saved Model (.pkl)       │
└──────────────────────────────┘
```

---

# 💾 Model Pipeline & Serialization

The final workflow creates a Scikit-learn pipeline containing:

```text
StandardScaler
      ↓
Tuned XGBoost
```

The pipeline is saved using `joblib`.

```python
joblib.dump(
    final_pipeline,
    "fraud_detection_model.pkl"
)
```

The saved pipeline can then be loaded:

```python
loaded_model = joblib.load(
    "fraud_detection_model.pkl"
)
```

The model generates fraud probabilities:

```python
sample_probs = loaded_model.predict_proba(
    X_test[:10]
)[:, 1]
```

and converts probabilities into fraud/legitimate labels using a chosen threshold.

Example:

```python
final_labels = (
    sample_probs >= 0.25
).astype(int)
```

---

# 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Credit Card Fraud Detection.ipynb
│
├── fraud_detection_model.pkl
│
├── Part_A_Professional_Answers_Fraud_Detection.pdf
│
└── README.md
```

---

# 🛠️ Tech Stack

### Programming Language

* Python

### Data Manipulation

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* Random Forest
* XGBoost

### Imbalanced Learning

* imbalanced-learn
* SMOTE
* Random Undersampling

### Model Selection

* RandomizedSearchCV
* Cross-Validation

### Model Persistence

* Joblib

### Development Environment

* Jupyter Notebook
* Git
* GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/dhorajiyamisri/Credit-Card-Fraud-Detection.git
```

Move into the project:

```bash
cd Credit-Card-Fraud-Detection
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Credit Card Fraud Detection.ipynb
```

---

# ▶️ How to Run

### Step 1

Place the dataset in the project directory:

```text
creditcard.csv
```

### Step 2

Open:

```text
Credit Card Fraud Detection.ipynb
```

### Step 3

Run the notebook cells sequentially.

The workflow performs:

```text
EDA
↓
Feature Engineering
↓
Scaling
↓
Train-Test Split
↓
SMOTE / Undersampling
↓
Model Training
↓
Model Evaluation
↓
XGBoost Tuning
↓
Threshold Optimization
↓
Pipeline Saving
```

---

# 🧠 Key Machine Learning Learnings

This project demonstrates several important real-world ML concepts.

### 1. Accuracy can be misleading

With extreme class imbalance, a model can have very high accuracy while failing to detect fraud.

---

### 2. Recall matters in fraud detection

Missing a fraudulent transaction can cause direct financial loss.

---

### 3. Precision also matters

Flagging too many legitimate transactions can increase manual investigation costs and negatively affect customer experience.

---

### 4. Resampling can improve minority-class learning

SMOTE and Random Undersampling provide different approaches to dealing with class imbalance.

---

### 5. PR-AUC is useful for rare-event classification

Precision-Recall analysis provides a more meaningful view of model behavior when the positive class is extremely rare.

---

### 6. Threshold selection is a business decision

A threshold of `0.5` is not automatically optimal.

The correct threshold depends on the relative cost of:

```text
False Positive
        vs
False Negative
```

---

### 7. Model performance is not only about the algorithm

The project demonstrates that:

```text
Data Preparation
        +
Imbalance Handling
        +
Model Selection
        +
Hyperparameter Tuning
        +
Threshold Optimization
        +
Business Cost
```

all contribute to the final fraud detection strategy.

---

# 💼 Real-World Applications

The approach demonstrated in this project can be adapted for:

* 🏦 Banks
* 💳 Credit Card Companies
* 💰 FinTech Platforms
* 🛒 E-commerce Payment Systems
* 💸 Digital Payment Platforms
* 🏢 Financial Institutions
* 🔐 Transaction Monitoring Systems

A production system could use transaction probabilities to trigger:

```text
Low Risk
   ↓
Approve Transaction

Medium Risk
   ↓
Additional Verification

High Risk
   ↓
Flag / Block / Investigate
```

---

# 🔮 Future Improvements

Possible future improvements include:

### 🚀 Model Improvements

* Advanced ensemble models
* LightGBM / CatBoost comparison
* Calibration of predicted probabilities
* Cost-sensitive learning

### ⚖️ Imbalance Handling

* Compare multiple SMOTE variants
* SMOTE + undersampling combinations
* Class-weight optimization

### 🎯 Threshold Optimization

Develop an automated threshold-selection system based on:

```text
Fraud Loss
+
Investigation Cost
+
Customer Impact
```

### 📊 Monitoring

A production system could monitor:

* Data drift
* Concept drift
* Fraud-rate changes
* Precision degradation
* Recall degradation

### 🌐 Production Deployment

The saved pipeline could eventually be integrated into:

```text
REST API
        ↓
Transaction
        ↓
ML Model
        ↓
Fraud Probability
        ↓
Risk Decision
```

---

# ⚠️ Limitations

This project is a machine learning prototype and should not be considered a production-ready banking fraud prevention system.

Real-world deployment would require:

* Secure transaction infrastructure
* Real-time inference
* Model monitoring
* Data drift detection
* Explainability
* Regulatory compliance
* Privacy and security controls
* Continuous retraining

The business simulation values such as transaction value and investigation cost are **project assumptions**, not real financial institution costs.

---

# 📌 Project Highlights

```text
✔ 284,807 transaction records
✔ Severe class imbalance analysis
✔ Feature engineering
✔ Log transformation
✔ Time-of-day feature extraction
✔ Standard scaling
✔ Stratified train-test split
✔ SMOTE oversampling
✔ Random undersampling
✔ Logistic Regression comparison
✔ Random Forest
✔ XGBoost
✔ RandomizedSearchCV
✔ 3-Fold Cross-Validation
✔ Precision-Recall analysis
✔ PR-AUC optimization
✔ Decision threshold optimization
✔ Business cost-benefit simulation
✔ Saved ML pipeline
✔ Joblib model serialization
```

---

# 🎤 Interview Explanation

### Short Version

> **"I developed a Credit Card Fraud Detection machine learning project as a binary classification problem. The main challenge was extreme class imbalance, where fraudulent transactions represented only around 0.17% of the dataset. I performed EDA, feature engineering, scaling and stratified train-test splitting. I experimented with SMOTE and Random Undersampling, then compared Logistic Regression, Random Forest and XGBoost. I further tuned XGBoost using RandomizedSearchCV with Average Precision as the scoring metric. Finally, I optimized the classification threshold based on F1 and recall requirements and connected the model with a cost-benefit analysis to understand the business impact of false positives and false negatives."**

---

# 🏆 What Makes This Project Strong?

This project goes beyond simply:

```text
Load Dataset
      ↓
Train Model
      ↓
Print Accuracy
```

Instead, it addresses the actual challenges of fraud detection:

```text
Extreme Class Imbalance
          ↓
Minority Class Learning
          ↓
Multiple ML Models
          ↓
Hyperparameter Optimization
          ↓
Precision-Recall Trade-off
          ↓
Threshold Optimization
          ↓
Financial Cost Analysis
          ↓
Reusable ML Pipeline
```

That makes the project more relevant to **real-world Data Science and Machine Learning workflows**.

---

# 👨‍💻 Author

<div align="center">

## Misari Dhorajiya

**Data Science / AI-ML Enthusiast**

Diploma in Information Technology
Government Polytechnic Gandhinagar

Currently pursuing **AI & ML with Data Science**

<br>

<a href="https://github.com/dhorajiyamisri">
  <img src="https://img.shields.io/badge/GitHub-dhorajiyamisri-black?style=for-the-badge&logo=github" />
</a>

</div>

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

**Built with Python • Machine Learning • Data Science**

</div>
