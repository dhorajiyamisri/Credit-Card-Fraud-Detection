<div align="center">

# 💳 Credit Card Fraud Detection

### 🛡️ An End-to-End Machine Learning System for Detecting Fraudulent Transactions

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Boosting-189FDD?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/SMOTE-Imbalanced%20Learning-EF4444?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Binary%20Classification-8B5CF6?style=flat-square"/>
  <img src="https://img.shields.io/badge/PR--AUC-Focused-22C55E?style=flat-square"/>
  <img src="https://img.shields.io/badge/Threshold%20Optimization-F59E0B?style=flat-square"/>
  <img src="https://img.shields.io/badge/Business%20Impact-06B6D4?style=flat-square"/>
</p>

<br>

### 🚨 284,807 Transactions • 0.17% Fraud • ML • Threshold Engineering • Business Analysis

<br>

<a href="https://github.com/dhorajiyamisri/Credit-Card-Fraud-Detection">
<img src="https://img.shields.io/badge/🔗%20VIEW%20PROJECT-111827?style=for-the-badge"/>
</a>

</div>

---

## 📌 Project Overview

**Credit Card Fraud Detection** is an end-to-end Machine Learning project that identifies whether a credit card transaction is **legitimate or potentially fraudulent**.

The project focuses on one of the biggest challenges in fraud analytics:

> ⚠️ **Fraudulent transactions represent only a tiny fraction of the complete dataset.**

Because of this extreme class imbalance, simply maximizing accuracy is not enough.

The project therefore combines:

* 🔎 Exploratory Data Analysis
* 🧬 Feature Engineering
* ⚖️ Imbalance Handling
* 🤖 Multiple ML Models
* 🔬 Hyperparameter Optimization
* 📈 Precision-Recall Analysis
* 🎚️ Threshold Optimization
* 💰 Business Impact Simulation
* 💾 Reusable ML Pipeline

---

# 🎯 Business Problem

Imagine a payment system processing thousands of transactions.

Most transactions are legitimate, but a small number may be fraudulent.

The model needs to answer:

> **"Should this transaction be treated as potentially fraudulent?"**

There are two major risks:

### 🔴 False Negative

A fraudulent transaction is classified as legitimate.

**Result → Potential financial loss**

### 🟠 False Positive

A legitimate transaction is flagged as fraudulent.

**Result → Unnecessary investigation and operational cost**

Therefore, the goal is not simply:

```text
❌ Maximum Accuracy
```

The real goal is:

```text
✅ Detect More Fraud
       +
✅ Control False Alerts
       +
✅ Minimize Business Cost
```

---

# 📊 Dataset

The project uses a credit-card transaction dataset containing:

| Property                   |        Value |
| -------------------------- | -----------: |
| 💳 Total Transactions      |  **284,807** |
| 📋 Total Columns           |       **31** |
| 🟢 Legitimate Transactions | **99.8273%** |
| 🔴 Fraudulent Transactions |  **0.1727%** |
| 🎯 Target Variable         |      `Class` |

### Target Variable

```text
Class = 0 → Legitimate Transaction
Class = 1 → Fraudulent Transaction
```

---

# ⚠️ The Core Challenge — Class Imbalance

The dataset is extremely imbalanced.

```text
Legitimate  ████████████████████████████████████████  99.8273%

Fraud       ▏                                         0.1727%
```

A model could achieve very high accuracy simply by predicting almost every transaction as legitimate.

That would make the model practically useless.

Therefore, this project focuses heavily on:

```text
🎯 Precision
🔍 Recall
⚖️ F1 Score
📈 PR-AUC
```

rather than accuracy alone.

---

# 🔎 Exploratory Data Analysis

Before training Machine Learning models, the dataset was explored to understand:

### 📊 Analysis Performed

* Transaction class distribution
* Fraud vs legitimate transactions
* Transaction amount distribution
* Feature behavior
* Minority-class characteristics
* Class imbalance visualization

The class distribution was also visualized using a logarithmic scale to make the extremely small fraud class easier to observe.

---

# 🧬 Feature Engineering

Two useful features were created from the original transaction information.

## 💰 Log Transaction Amount

```python
df['Amount_log'] = np.log1p(df['Amount'])
```

### Why?

Transaction amounts can be highly skewed.

The logarithmic transformation reduces the influence of extreme values.

---

## 🕐 Transaction Hour

```python
df['Hour'] = (df['Time'] % 86400) // 3600
```

This extracts the transaction hour and allows the model to learn possible time-based patterns.

---

## 🗑️ Original Features Removed

After creating the engineered features:

```python
df = df.drop(['Time', 'Amount'], axis=1)
```

---

# 📐 Feature Scaling

`StandardScaler` was applied to the engineered numerical features.

```python
scaler = StandardScaler()

df[['Amount_log', 'Hour']] = scaler.fit_transform(
    df[['Amount_log', 'Hour']]
)
```

This standardizes the feature scale and is especially useful for algorithms such as Logistic Regression.

---

# ✂️ Train-Test Split

The dataset was divided into:

```text
🟦 80% → Training Data
🟨 20% → Testing Data
```

Stratified splitting was used:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)
```

### Why Stratification?

Because fraud is extremely rare, stratification helps preserve the class distribution in both training and testing datasets.

---

# ⚖️ Imbalance Handling

Two major approaches were experimented with.

```text
                 CLASS IMBALANCE
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
       🔵 SMOTE              🟠 Under-Sampling
          │                         │
 Generate minority data       Reduce majority data
```

---

## 🔵 SMOTE

**SMOTE — Synthetic Minority Over-sampling Technique**

Instead of simply duplicating fraud records, SMOTE generates synthetic minority-class examples.

### Training Data

Before SMOTE:

```text
🟢 Legitimate → 227,451
🔴 Fraud      → 394
```

After SMOTE:

```text
🟢 Legitimate → 227,451
🔴 Fraud      → 22,745
```

---

## 🟠 Random Under-Sampling

Under-sampling reduces the number of majority-class observations.

After under-sampling:

```text
🟢 Legitimate → 3,940
🔴 Fraud      → 394
```

### Comparison

| Technique         | Approach                             | Benefit                         |
| ----------------- | ------------------------------------ | ------------------------------- |
| 🔵 SMOTE          | Generates synthetic minority samples | Retains majority information    |
| 🟠 Under-Sampling | Removes majority samples             | Faster and smaller training set |

---

# 🤖 Model Experimentation

Instead of relying on a single algorithm, multiple Machine Learning approaches were compared.

```text
             🤖 MODEL EXPERIMENTATION
                       │
        ┌──────────────┼──────────────┐
        ↓              ↓              ↓
   Logistic        Random         XGBoost
  Regression       Forest
        │              │              │
        └──────────────┼──────────────┘
                       ↓
                Model Evaluation
                       ↓
              Hyperparameter Tuning
                       ↓
              Threshold Optimization
```

---

# 1️⃣ Logistic Regression

Logistic Regression was used as the baseline model.

Three strategies were tested:

* Original data + balanced class weights
* SMOTE
* Random Under-Sampling

### Results

| Approach            | Precision |     Recall |         F1 |     PR-AUC |
| ------------------- | --------: | ---------: | ---------: | ---------: |
| Original + Balanced |    0.0555 | **0.9082** |     0.1046 |     0.7128 |
| SMOTE               |    0.3686 |     0.8878 | **0.5210** |     0.7418 |
| Under-Sampling      |    0.3372 |     0.8878 |     0.4888 | **0.7449** |

### 💡 Observation

The balanced Logistic Regression model achieved very high recall, but its precision was low.

This means many legitimate transactions could be incorrectly flagged as fraud.

---

# 2️⃣ Random Forest 🌲

Random Forest was trained using SMOTE-resampled data.

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### Results

```text
🎯 Precision → 0.8367
🔍 Recall    → 0.8367
⚖️ F1 Score  → 0.8367
📈 PR-AUC    → 0.8536
```

Random Forest provided a strong balance between precision and recall.

---

# 3️⃣ XGBoost ⚡

XGBoost was introduced as a powerful gradient-boosting model for the transaction data.

Class imbalance was incorporated using:

```python
scale_pos_weight
```

The baseline model used:

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

### Baseline Results

```text
🎯 Precision → 0.6563
🔍 Recall    → 0.8571
⚖️ F1 Score  → 0.7434
📈 PR-AUC    → 0.8579
```

---

# 🏆 Model Performance

| Model                                |  Precision |     Recall |         F1 |     PR-AUC |
| ------------------------------------ | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression — Balanced       |     0.0555 |     0.9082 |     0.1046 |     0.7128 |
| Logistic Regression — SMOTE          |     0.3686 |     0.8878 |     0.5210 |     0.7418 |
| Logistic Regression — Under-Sampling |     0.3372 |     0.8878 |     0.4888 |     0.7449 |
| 🌲 Random Forest + SMOTE             | **0.8367** | **0.8367** | **0.8367** | **0.8536** |
| ⚡ XGBoost Baseline                   |     0.6563 | **0.8571** |     0.7434 | **0.8579** |
| ⚡ Tuned XGBoost                      |          — |          — |          — | **0.8541** |

> 📌 Metrics are reported from the corresponding notebook experiments.

---

# 🔬 Hyperparameter Optimization

The XGBoost model was further optimized using:

### `RandomizedSearchCV`

Search parameters included:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
```

### Configuration

```text
🔁 Cross Validation → 3-Fold
🎯 Scoring → Average Precision
🔬 Random Search Iterations → 15
```

---

# 🥇 Best XGBoost Configuration

The selected configuration was:

| Parameter          |   Value |
| ------------------ | ------: |
| `n_estimators`     | **300** |
| `max_depth`        |   **3** |
| `learning_rate`    | **0.1** |
| `subsample`        | **0.6** |
| `colsample_bytree` | **0.8** |

### Performance

```text
🏆 Best CV PR-AUC  → 0.842884
📈 Test PR-AUC     → 0.854052
```

---

# 📈 Why PR-AUC?

For heavily imbalanced classification, PR-AUC is especially useful because it focuses on the relationship between:

```text
Precision ↔ Recall
```

The project therefore uses **Average Precision / PR-AUC** as an important model-selection metric.

---

# 🎚️ Threshold Optimization

Machine Learning models generate probabilities.

The default decision threshold of `0.50` is not always the best choice.

This project investigates different thresholds based on:

```text
🎯 F1 Score
🔍 Recall
💰 Business Cost
```

---

## 🏅 F1-Optimal Threshold

The threshold that maximized F1 in the experiment was approximately:

```text
🎯 0.9747
```

---

## 🚨 High-Recall Threshold

The project also searched for a threshold achieving at least 90% recall.

The resulting threshold was approximately:

```text
🎯 0.1114
```

### Business Interpretation

```text
Lower Threshold
      ↓
More Fraud Alerts
      ↓
Higher Recall
      ↓
Potentially More False Positives
```

So threshold selection should depend on the business objective.

---

# 🚦 Risk Decision Framework

A fraud probability can be converted into a risk category.

### Illustrative Example

```text
🟢 0.00 – 0.20 → Low Risk
🟡 0.20 – 0.50 → Medium Risk
🟠 0.50 – 0.80 → High Risk
🔴 0.80 – 1.00 → Critical Risk
```

> ⚠️ These risk bands are illustrative project categories and are not real banking thresholds.

A real production system would determine thresholds using historical fraud cost, customer impact and investigation capacity.

---

# 💰 Business Impact Simulation

The project also connects Machine Learning predictions with a simplified financial model.

### Assumptions

```text
💳 Average Fraud Transaction = ₹4,500
🔎 Investigation Cost         = ₹150
```

---

### 💰 Money Saved

```text
Money Saved
=
True Positives × ₹4,500
```

### 🔎 Investigation Cost

```text
Investigation Cost
=
(True Positives + False Positives) × ₹150
```

### 💸 Money Lost

```text
Money Lost
=
False Negatives × ₹4,500
```

### 🧮 Net Benefit

```text
Net Benefit
=
Money Saved − Investigation Cost
```

This allows the model to be evaluated from a **business perspective**, not only a Machine Learning perspective.

> 📌 These values are assumptions used for project-level simulation.

---

# 🧠 Key Insight

The project demonstrates an important Data Science principle:

> ### **The best ML model is not always the model with the highest accuracy.**

For fraud detection, we need to balance:

```text
                 MODEL QUALITY
                      │
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
     Precision      Recall        PR-AUC
        │             │             │
        └─────────────┼─────────────┘
                      ↓
              Decision Threshold
                      ↓
               Business Impact
```

---

# 💾 Final ML Pipeline

The final workflow is:

```text
💳 Transaction
      ↓
🧬 Feature Engineering
      ↓
📐 Scaling
      ↓
⚡ Tuned XGBoost
      ↓
📊 Fraud Probability
      ↓
🎚️ Decision Threshold
      ↓
🟢 Legitimate / 🔴 Fraud
```

The trained pipeline was serialized using Joblib:

```python
joblib.dump(
    final_pipeline,
    "fraud_detection_model.pkl"
)
```

It can be loaded later using:

```python
loaded_model = joblib.load(
    "fraud_detection_model.pkl"
)
```

---

# 🏗️ Project Architecture

```text
                         💳 TRANSACTION
                               │
                               ▼
                     ┌──────────────────┐
                     │   Data Loading   │
                     └────────┬─────────┘
                              ↓
                     ┌──────────────────┐
                     │       EDA        │
                     └────────┬─────────┘
                              ↓
                     ┌──────────────────┐
                     │ Feature Engineer │
                     └────────┬─────────┘
                              ↓
                     ┌──────────────────┐
                     │     Scaling      │
                     └────────┬─────────┘
                              ↓
                     ┌──────────────────┐
                     │ Imbalance Handle │
                     └────────┬─────────┘
                              ↓
                     ┌──────────────────┐
                     │  Model Training  │
                     └────────┬─────────┘
                              ↓
                     ┌──────────────────┐
                     │ Hyperparameter   │
                     │     Tuning       │
                     └────────┬─────────┘
                              ↓
                     ┌──────────────────┐
                     │ Threshold Engine │
                     └────────┬─────────┘
                              ↓
                     ┌──────────────────┐
                     │ Business Impact  │
                     └────────┬─────────┘
                              ↓
                     🛡️ FRAUD DECISION
```

---

# 📁 Repository Structure

```text
Credit-Card-Fraud-Detection/
│
├── 📓 Credit Card Fraud Detection.ipynb
│
├── 💾 fraud_detection_model.pkl
│
├── 📄 Part_A_Professional_Answers_Fraud_Detection.pdf
│
└── 📘 README.md
```

---

# 🛠️ Technology Stack

### Programming & Data

* 🐍 Python
* 📊 Pandas
* 🔢 NumPy

### Visualization

* 📈 Matplotlib
* 📊 Seaborn

### Machine Learning

* 🤖 Scikit-learn
* 🌲 Random Forest
* ⚡ XGBoost
* 📉 Logistic Regression

### Imbalanced Learning

* ⚖️ imbalanced-learn
* 🔵 SMOTE
* 🟠 Random Under-Sampling

### Optimization

* 🔬 RandomizedSearchCV
* 🔁 Cross-Validation

### Model Management

* 💾 Joblib

### Development

* 📓 Jupyter Notebook
* 🌐 GitHub
* 🔧 Git

---

# 🚀 Installation & Usage

## 1. Clone the Repository

```bash
git clone https://github.com/dhorajiyamisri/Credit-Card-Fraud-Detection.git
```

## 2. Navigate to the Project

```bash
cd Credit-Card-Fraud-Detection
```

## 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib jupyter
```

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Credit Card Fraud Detection.ipynb
```

---

# ▶️ Project Workflow

```text
01. Load Dataset
        ↓
02. Understand Data
        ↓
03. Analyze Fraud Distribution
        ↓
04. Feature Engineering
        ↓
05. Feature Scaling
        ↓
06. Train-Test Split
        ↓
07. SMOTE / Under-Sampling
        ↓
08. Train ML Models
        ↓
09. Compare Performance
        ↓
10. Tune XGBoost
        ↓
11. Analyze Precision-Recall
        ↓
12. Optimize Threshold
        ↓
13. Simulate Business Impact
        ↓
14. Save Final Pipeline
```

---

# 💡 Key Learnings

### 01. Accuracy Can Be Misleading

Highly imbalanced datasets can make accuracy look impressive even when fraud detection is poor.

### 02. Recall Matters

Missing an actual fraudulent transaction can lead to financial loss.

### 03. Precision Matters

Too many false positives increase investigation workload.

### 04. Resampling Changes Model Behavior

SMOTE and under-sampling affect how the model learns the minority class.

### 05. Threshold Is a Business Decision

The probability threshold can be changed depending on the cost of false positives and false negatives.

### 06. Model Performance ≠ Business Performance

A model should ultimately be evaluated based on the decisions and costs it creates.

---

# 🌍 Real-World Applications

This approach can be adapted for:

```text
🏦 Banking
💳 Credit Card Processing
💸 FinTech
🛒 E-Commerce Payments
📱 Digital Wallets
🔐 Transaction Monitoring
🏢 Financial Institutions
```

A production system could follow:

```text
Transaction
     ↓
Risk Score
     ↓
┌────────────┬─────────────┬──────────────┐
│ 🟢 Low     │ 🟡 Medium   │ 🔴 High      │
│ Risk       │ Risk        │ Risk         │
├────────────┼─────────────┼──────────────┤
│ Approve    │ Verify      │ Investigate  │
└────────────┴─────────────┴──────────────┘
```

---

# 🔮 Future Improvements

### 🤖 Machine Learning

* LightGBM comparison
* CatBoost comparison
* Ensemble learning
* Probability calibration
* Cost-sensitive learning

### 🎯 Decision System

* Automated threshold optimization
* Cost-sensitive threshold selection
* Dynamic risk scoring
* Explainable fraud decisions

### 📊 Production

* Real-time prediction API
* Model monitoring
* Data drift detection
* Concept drift detection
* Automated retraining

### 🔐 Enterprise

* Secure deployment
* Transaction-level monitoring
* Fraud investigation workflow
* Privacy and compliance controls

---

# ⚠️ Limitations

This project is a **Machine Learning portfolio prototype** and not a production banking fraud-prevention system.

A real-world implementation would require:

* 🔐 Secure infrastructure
* ⚡ Real-time inference
* 📊 Monitoring
* 🔄 Data drift detection
* 🧠 Explainability
* 🔒 Privacy controls
* ⚖️ Regulatory compliance
* 🚨 Fraud investigation workflow
* 🔁 Continuous retraining

---

# 🎤 Interview Explanation

> **"I developed an end-to-end Credit Card Fraud Detection system using Machine Learning. The dataset contained 284,807 transactions with only 0.17% fraud, making class imbalance the main challenge. I performed EDA and feature engineering, used SMOTE and under-sampling, and compared Logistic Regression, Random Forest and XGBoost. I then tuned XGBoost using RandomizedSearchCV with Average Precision as the scoring metric. Finally, I performed Precision-Recall and threshold optimization along with a business impact simulation, and saved the final prediction pipeline using Joblib."**

---

# 🏆 Project Highlights

<div align="center">

### 💳 284K+ Transactions

### ⚠️ 0.17% Fraud

### 🤖 Multiple ML Models

### ⚖️ Imbalance Handling

### 🔬 Hyperparameter Tuning

### 📈 PR-AUC Evaluation

### 🎚️ Threshold Optimization

### 💰 Business Impact Analysis

### 💾 Reusable ML Pipeline

</div>

---

# 🌟 Why This Project Stands Out

Most beginner fraud-detection projects stop at:

```text
Dataset
   ↓
Model
   ↓
Accuracy
```

This project goes further:

```text
Dataset
   ↓
EDA
   ↓
Feature Engineering
   ↓
Imbalance Handling
   ↓
Multiple Models
   ↓
Hyperparameter Tuning
   ↓
Precision-Recall Analysis
   ↓
Threshold Optimization
   ↓
Business Cost Simulation
   ↓
Reusable ML Pipeline
```

### 🚀 The focus is not only:

> **"Can Machine Learning detect fraud?"**

### It is also:

> **"How should the model's prediction be converted into a practical business decision?"**

---

<div align="center">

# 💳 Detect Fraud. Reduce Risk. Make Better Decisions.

### Built with 🐍 Python • 🤖 Machine Learning • 📊 Data Science

<br>

<a href="https://github.com/dhorajiyamisri/Credit-Card-Fraud-Detection">
<img src="https://img.shields.io/badge/⭐%20STAR%20THIS%20PROJECT-111827?style=for-the-badge"/>
</a>

<br><br>

**Thank you for visiting this project! 🚀**

</div>
