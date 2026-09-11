<div align="center">

# 💳⚡ Credit Card Fraud Detection

### 🛡️ Intelligent Machine Learning for Detecting Suspicious Transactions

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Boosting-189FDD?style=for-the-badge&logo=xgboost&logoColor=white"/>
  <img src="https://img.shields.io/badge/SMOTE-Imbalanced%20Learning-EF4444?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Classification-Binary-8B5CF6?style=flat-square"/>
  <img src="https://img.shields.io/badge/PR--AUC-Focused-22C55E?style=flat-square"/>
  <img src="https://img.shields.io/badge/Threshold-Optimization-F59E0B?style=flat-square"/>
  <img src="https://img.shields.io/badge/Business-Driven-06B6D4?style=flat-square"/>
</p>

<br>

<a href="https://github.com/dhorajiyamisri/Credit-Card-Fraud-Detection">
<img src="https://img.shields.io/badge/🔗%20VIEW%20PROJECT-GitHub-111827?style=for-the-badge"/>
</a>

</div>

---

# 🚨 Project Snapshot

> **Credit Card Fraud Detection** is an end-to-end Machine Learning project that identifies suspicious credit card transactions under an extremely imbalanced classification environment.

Instead of depending only on **Accuracy**, this project focuses on the metrics and decisions that actually matter in fraud detection:

```text
🎯 Precision
🎯 Recall
🎯 F1 Score
🎯 PR-AUC
🎯 Threshold Optimization
🎯 Financial Cost Analysis
```

The project explores multiple approaches for handling class imbalance and compares:

```text
Logistic Regression
        ↓
Random Forest
        ↓
XGBoost
        ↓
Hyperparameter Tuning
        ↓
Threshold Optimization
        ↓
Business Decision
```

---

# 🧠 What Makes This Project Different?

Most beginner fraud-detection projects follow:

```text
Dataset
   ↓
Train Model
   ↓
Accuracy
   ↓
Done ❌
```

This project follows a more realistic workflow:

```text
📥 Transaction Data
        ↓
🔎 Exploratory Analysis
        ↓
🧹 Feature Engineering
        ↓
⚖️ Imbalance Analysis
        ↓
🔄 SMOTE / Undersampling
        ↓
🤖 Multiple ML Models
        ↓
🎯 Hyperparameter Tuning
        ↓
📈 Precision-Recall Analysis
        ↓
🎚️ Threshold Optimization
        ↓
💰 Cost-Benefit Analysis
        ↓
💾 Reusable ML Pipeline
```

> **Fraud detection is not just a classification problem — it is a decision-making problem under financial risk.**

---

# 🎯 Business Problem

Credit card companies process a huge number of legitimate transactions every day, while fraudulent transactions represent only a tiny fraction.

The objective is:

> 🔐 **Identify potentially fraudulent transactions while minimizing unnecessary investigation of legitimate transactions.**

This creates a difficult trade-off.

### 🚨 False Negative

A fraudulent transaction is predicted as legitimate.

```text
Fraud
  ↓
Model says Legitimate
  ↓
Fraud goes undetected
  ↓
💰 Financial Loss
```

### ⚠️ False Positive

A legitimate transaction is predicted as fraud.

```text
Legitimate
  ↓
Model says Fraud
  ↓
Investigation Required
  ↓
💰 Investigation Cost
```

Therefore:

> **The best model is not necessarily the model with the highest accuracy.**

---

# 📊 Dataset

The project works with a credit-card transaction dataset containing:

```text
📦 284,807 Transactions
📋 31 Columns
🎯 Binary Target → Class
```

### Target Variable

| Value | Meaning                   |
| ----: | ------------------------- |
|   `0` | 🟢 Legitimate Transaction |
|   `1` | 🔴 Fraudulent Transaction |

### Class Distribution

```text
🟢 Legitimate → 99.8273%
🔴 Fraud      → 0.1727%
```

This extreme imbalance is one of the central challenges of the project.

---

# ⚠️ The Class Imbalance Challenge

Imagine a system with:

```text
100,000 transactions
```

If only around 170 are fraudulent, a model could predict almost everything as legitimate and still appear highly accurate.

That is why:

```text
                 ❌ Accuracy Alone
                       ↓
                Misleading Result
```

Instead, this project prioritizes:

```text
                 ✅ Precision
                 ✅ Recall
                 ✅ F1 Score
                 ✅ PR-AUC
```

---

# 🔎 Exploratory Data Analysis

The project performs multiple EDA steps before model training.

### 📌 Class Distribution

The transaction classes are visualized using a logarithmic scale to clearly display the huge imbalance.

### 💰 Transaction Amount Analysis

Transaction amounts are compared between:

```text
🟢 Legitimate Transactions
🔴 Fraudulent Transactions
```

This helps understand whether transaction-value behavior differs across the two classes.

---

# 🧹 Feature Engineering

The project performs feature engineering before model training.

## 💰 Amount Transformation

The original transaction amount is transformed using:

```python
df['Amount_log'] = np.log1p(df['Amount'])
```

### Why?

Transaction amount can be highly skewed.

Log transformation helps reduce the impact of extreme values and produces a more manageable feature distribution.

---

## 🕐 Transaction Hour

The original `Time` feature is converted into an hourly feature:

```python
df['Hour'] = (df['Time'] % 86400) // 3600
```

This allows the model to capture possible time-of-day patterns.

---

## 🗑️ Original Feature Removal

After feature engineering:

```python
df = df.drop(['Time','Amount'], axis=1)
```

The transformed representations are retained instead.

---

# 📐 Feature Scaling

The engineered features are standardized using:

```python
StandardScaler()
```

Example:

```python
scaler = StandardScaler()

df[['Amount_log','Hour']] = scaler.fit_transform(
    df[['Amount_log','Hour']]
)
```

This places the transformed features on a comparable scale.

---

# ✂️ Train-Test Split

The dataset is divided using an:

```text
80% → Training
20% → Testing
```

with:

```python
stratify=y
```

### Why Stratification?

Because fraud transactions are extremely rare, maintaining the class distribution between training and testing sets is important.

---

# ⚖️ Imbalance Handling

The project experiments with **two different resampling strategies**.

---

## 🔵 01 — SMOTE

### Synthetic Minority Over-sampling Technique

SMOTE generates synthetic examples for the minority class.

```text
Before SMOTE

🟢 Majority → 227,451
🔴 Fraud    → 394
```

After SMOTE:

```text
🟢 Majority → 227,451
🔴 Fraud    → 22,745
```

The goal is to give the model more minority-class information during training.

---

## 🟠 02 — Random Undersampling

Random Undersampling reduces the number of majority-class samples.

```text
Before

🟢 Majority → 227,451
🔴 Fraud    → 394
```

After:

```text
🟢 Majority → 3,940
🔴 Fraud    → 394
```

### Comparison

| Method           | Main Idea                      | Advantage                       |
| ---------------- | ------------------------------ | ------------------------------- |
| 🔵 SMOTE         | Create synthetic fraud samples | Retains majority information    |
| 🟠 Undersampling | Reduce majority samples        | Faster and smaller training set |

---

# 🤖 Machine Learning Lab

The project experiments with multiple models rather than depending on a single algorithm.

```text
                    🧪 MODEL LAB
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     Logistic        Random         XGBoost
    Regression       Forest
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                 🏆 Model Evaluation
                         ↓
                🎯 Hyperparameter Tuning
                         ↓
                🎚️ Threshold Optimization
```

---

# 1️⃣ Logistic Regression

Logistic Regression is used as the baseline classification model.

Three approaches are evaluated:

```text
① Original Data + Balanced Class Weight
② SMOTE
③ Random Undersampling
```

### 📊 Results

| Strategy                   | Precision | Recall |     F1 | PR-AUC |
| -------------------------- | --------: | -----: | -----: | -----: |
| Original + Balanced Weight |    0.0555 | 0.9082 | 0.1046 | 0.7128 |
| SMOTE                      |    0.3686 | 0.8878 | 0.5210 | 0.7418 |
| Undersampling              |    0.3372 | 0.8878 | 0.4888 | 0.7449 |

### 💡 Insight

The baseline achieves strong recall but suffers from low precision.

SMOTE improves the balance between precision and recall considerably.

---

# 2️⃣ Random Forest 🌲🌲🌲

Random Forest is trained using SMOTE-resampled data.

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### 📈 Verified Results

```text
🎯 Precision → 0.8367
🎯 Recall    → 0.8367
🎯 F1 Score  → 0.8367
🎯 PR-AUC    → 0.8536
```

Random Forest also provides feature importance analysis, helping identify influential transaction variables.

---

# 3️⃣ XGBoost ⚡

XGBoost is introduced as a stronger gradient boosting approach.

The project calculates:

```python
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]
```

and uses:

```python
scale_pos_weight = ratio
```

to account for the class imbalance.

### ⚙️ Baseline Configuration

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

### 📊 Verified Results

```text
Precision → 0.65625
Recall    → 0.85714
F1 Score  → 0.74336
PR-AUC    → 0.85793
```

---

# 🎯 Hyperparameter Optimization

The XGBoost model is further optimized using:

```text
🔬 RandomizedSearchCV
```

### Parameters explored

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

```text
🔁 3-Fold Cross Validation
🎯 Scoring → Average Precision
🔬 15 Random Parameter Combinations
```

---

# 🏆 Best XGBoost Configuration

The notebook identifies:

```text
n_estimators       → 300
max_depth          → 3
learning_rate      → 0.1
subsample          → 0.6
colsample_bytree   → 0.8
```

### Cross-Validation

```text
🏆 Best CV PR-AUC
0.842884
```

### Test Performance

```text
📈 Tuned XGBoost PR-AUC
0.854052
```

---

# 📊 Model Performance Board

| 🧠 Model                            | 🎯 Precision |  🔍 Recall |      ⚖️ F1 |  📈 PR-AUC |
| ----------------------------------- | -----------: | ---------: | ---------: | ---------: |
| Logistic Regression — Original      |       0.0555 |     0.9082 |     0.1046 |     0.7128 |
| Logistic Regression — SMOTE         |       0.3686 |     0.8878 |     0.5210 |     0.7418 |
| Logistic Regression — Undersampling |       0.3372 |     0.8878 |     0.4888 |     0.7449 |
| 🌲 Random Forest + SMOTE            |   **0.8367** | **0.8367** | **0.8367** | **0.8536** |
| ⚡ XGBoost Baseline                  |       0.6563 | **0.8571** |     0.7434 | **0.8579** |
| ⚡ Tuned XGBoost                     |            — |          — |          — | **0.8541** |

> 📌 Metrics above use the exact values printed by the corresponding notebook model cells. Some later notebook comparison tables contain approximate (`~`) values, so those are not presented as exact here.

---

# 🎚️ Threshold Optimization

One of the most interesting parts of this project is **decision-threshold optimization**.

Most classification models use:

```text
Probability ≥ 0.50
        ↓
Fraud
```

But fraud detection does not necessarily require a fixed threshold of `0.50`.

The project investigates different thresholds and studies their effect on:

```text
🎯 Precision
🔍 Recall
⚖️ F1 Score
💰 Financial Benefit
```

---

# 🏅 F1-Optimal Threshold

The notebook searches for the threshold that maximizes F1 Score.

```text
🎯 Best Threshold ≈ 0.9747
```

This demonstrates that the optimal operating point can be very different from the default `0.5`.

---

# 🚨 High-Recall Strategy

The project also searches for a threshold capable of achieving at least:

```text
Recall ≥ 90%
```

The identified threshold is approximately:

```text
🎯 Threshold ≈ 0.1114
```

This creates a different operating strategy where the system prioritizes catching more fraudulent transactions.

---

# 🚦 Risk Decision Engine

The model probability can conceptually be converted into a risk score:

```text
             MODEL PROBABILITY
                     ↓
              ┌──────┴──────┐
              ↓             ↓
           LOW RISK      HIGH RISK
              ↓             ↓
           Approve       Investigate
```

For a production system, thresholds would be selected using historical fraud costs, customer impact and operational investigation capacity.

> ⚠️ The following risk bands are illustrative project-level categories, not real banking thresholds.

```text
🟢 0.00 – 0.20 → Low Risk
🟡 0.20 – 0.50 → Medium Risk
🟠 0.50 – 0.80 → High Risk
🔴 0.80 – 1.00 → Critical Risk
```

---

# 📈 Precision-Recall Analysis

The project uses Precision-Recall curves to understand model behavior across thresholds.

### Why PR Curve?

When the positive class is extremely rare, Precision-Recall analysis can provide more useful information than simply looking at accuracy.

```text
Precision
   ↑
   │\
   │ \
   │  \
   │   \
   │    \____
   │
   └──────────────→ Recall
```

The model can therefore be evaluated at multiple operating points rather than one arbitrary threshold.

---

# 💰 Financial Impact Engine

A major business-oriented component of the project is cost-benefit analysis.

### Project assumptions

```text
💳 Average Fraudulent Transaction = ₹4,500
🔎 Investigation Cost = ₹150
```

### Money Saved

```text
Money Saved
= True Positives × ₹4,500
```

### Investigation Cost

```text
Investigation Cost
= (True Positives + False Positives) × ₹150
```

### Money Lost

```text
Money Lost
= False Negatives × ₹4,500
```

### Net Benefit

```text
💰 Net Benefit
= Money Saved − Investigation Cost
```

---

# 🧮 Why Business Cost Matters

Imagine two models:

```text
Model A
High Recall
Many False Positives
        ↓
More investigations
```

versus:

```text
Model B
Slightly Lower Recall
Fewer False Positives
        ↓
Lower investigation cost
```

Which model is better?

> **It depends on the business cost of each type of mistake.**

That is why this project evaluates both **ML performance and financial impact**.

---

# 🧪 Experiment Tracking

The project can be viewed as a sequence of controlled experiments:

| 🧪 Experiment | ⚖️ Sampling       | 🤖 Model            | 🎯 Objective                |
| ------------- | ----------------- | ------------------- | --------------------------- |
| E1            | Original          | Logistic Regression | Baseline                    |
| E2            | SMOTE             | Logistic Regression | Minority learning           |
| E3            | Undersampling     | Logistic Regression | Majority reduction          |
| E4            | SMOTE             | Random Forest       | Non-linear learning         |
| E5            | Class Weight      | XGBoost             | Imbalance-aware boosting    |
| E6            | Tuned             | XGBoost             | Hyperparameter optimization |
| E7            | Tuned + Threshold | XGBoost             | Decision optimization       |

This makes the project an **ML experimentation workflow**, rather than a single-model notebook.

---

# 🔬 Model Selection Philosophy

| Model                      | Why It Was Used                               |
| -------------------------- | --------------------------------------------- |
| 📉 Logistic Regression     | Baseline + interpretable linear classifier    |
| 🌲 Random Forest           | Non-linear relationships + feature importance |
| ⚡ XGBoost                  | Powerful gradient boosting for tabular data   |
| 🔍 RandomizedSearchCV      | Hyperparameter optimization                   |
| 🎚️ Threshold Optimization | Business-aware decision making                |

---

# 💾 Production-Ready Pipeline

The final workflow creates a reusable ML pipeline:

```text
Input Transaction
       ↓
StandardScaler
       ↓
Tuned XGBoost
       ↓
Fraud Probability
       ↓
Decision Threshold
       ↓
Fraud / Legitimate
```

The pipeline is serialized using:

```python
joblib.dump(
    final_pipeline,
    "fraud_detection_model.pkl"
)
```

It can later be loaded using:

```python
loaded_model = joblib.load(
    "fraud_detection_model.pkl"
)
```

---

# 🧪 Prediction Flow

A new transaction can be processed as:

```text
New Transaction
      ↓
Feature Preparation
      ↓
Scaling
      ↓
XGBoost Prediction
      ↓
Fraud Probability
      ↓
Threshold
      ↓
┌─────────────────┐
│ Legitimate      │
│ OR              │
│ Fraud           │
└─────────────────┘
```

Example:

```python
sample_probs = loaded_model.predict_proba(
    X_test[:10]
)[:, 1]

final_labels = (
    sample_probs >= 0.25
).astype(int)
```

---

# 🏗️ End-to-End Architecture

```text
                         💳 TRANSACTION
                              │
                              ▼
                    ┌──────────────────┐
                    │ Data Validation  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Feature Engineer │
                    │ Amount_log       │
                    │ Hour             │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Feature Scaling  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │   XGBoost Model  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Fraud Probability│
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │  Risk Decision   │
                    └───────┬──────────┘
                            / \
                           /   \
                          ▼     ▼
                    🟢 Legit   🔴 Fraud
                             
```

---

# 📁 Project Structure

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

<div align="center">

| Category           | Technologies            |
| ------------------ | ----------------------- |
| 🐍 Programming     | Python                  |
| 📊 Data            | Pandas, NumPy           |
| 📈 Visualization   | Matplotlib, Seaborn     |
| 🤖 ML              | Scikit-learn            |
| ⚡ Boosting         | XGBoost                 |
| ⚖️ Imbalance       | imbalanced-learn, SMOTE |
| 🔍 Tuning          | RandomizedSearchCV      |
| 🔄 Validation      | Cross-Validation        |
| 💾 Serialization   | Joblib                  |
| 📓 Environment     | Jupyter Notebook        |
| 🌐 Version Control | Git, GitHub             |

</div>

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/dhorajiyamisri/Credit-Card-Fraud-Detection.git
```

Navigate into the project:

```bash
cd Credit-Card-Fraud-Detection
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib jupyter
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
📓 Credit Card Fraud Detection.ipynb
```

---

# ▶️ Project Workflow

Run the notebook sequentially:

```text
1️⃣ Load Dataset
        ↓
2️⃣ Explore Data
        ↓
3️⃣ Analyze Fraud Distribution
        ↓
4️⃣ Feature Engineering
        ↓
5️⃣ Feature Scaling
        ↓
6️⃣ Train-Test Split
        ↓
7️⃣ SMOTE / Undersampling
        ↓
8️⃣ Train ML Models
        ↓
9️⃣ Evaluate Models
        ↓
🔟 Tune XGBoost
        ↓
1️⃣1️⃣ Optimize Threshold
        ↓
1️⃣2️⃣ Business Cost Analysis
        ↓
1️⃣3️⃣ Save Pipeline
```

---

# 💡 Key Insights

### 🔹 Insight 01 — Accuracy Isn't Enough

Extreme class imbalance makes accuracy a poor standalone metric.

### 🔹 Insight 02 — Recall Has Financial Importance

A false negative means a potentially fraudulent transaction was missed.

### 🔹 Insight 03 — Precision Controls Operational Cost

A very aggressive fraud detector can generate too many false alarms.

### 🔹 Insight 04 — Resampling Changes Model Behavior

SMOTE and undersampling create different training environments.

### 🔹 Insight 05 — XGBoost Provides Strong Performance

The XGBoost experiments achieved strong PR-AUC while maintaining high fraud recall.

### 🔹 Insight 06 — Threshold Is a Business Lever

Changing the threshold changes the balance between fraud detection and investigation workload.

### 🔹 Insight 07 — Model ≠ Complete Solution

A production fraud system needs:

```text
Model
+
Threshold
+
Business Rules
+
Monitoring
+
Human Investigation
```

---

# 🌍 Real-World Applications

This type of system can support:

```text
🏦 Banking
💳 Credit Card Processing
💸 FinTech
🛒 E-Commerce Payments
📱 Digital Wallets
🏢 Financial Institutions
🔐 Transaction Monitoring
```

A production workflow could look like:

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

## 🚀 Model Improvements

* LightGBM comparison
* CatBoost comparison
* Ensemble learning
* Probability calibration
* Cost-sensitive optimization

## ⚖️ Imbalance Improvements

* Advanced SMOTE variants
* SMOTE + undersampling
* Class-weight optimization
* Cost-sensitive learning

## 🎯 Decision Improvements

Build an automated threshold engine based on:

```text
Fraud Loss
+
Investigation Cost
+
Customer Experience
+
Operational Capacity
```

## 📊 Monitoring

A production system could monitor:

```text
📈 Fraud Rate
📉 Precision Drift
📉 Recall Drift
🔄 Data Drift
🧠 Concept Drift
```

## 🌐 Deployment

Future architecture:

```text
Payment System
      ↓
REST API
      ↓
ML Pipeline
      ↓
Fraud Probability
      ↓
Risk Engine
      ↓
Business Decision
```

---

# 🛡️ Production Readiness

| Component                  | Status |
| -------------------------- | ------ |
| 📊 Data Analysis           | ✅      |
| 🧹 Preprocessing           | ✅      |
| ⚖️ Imbalance Handling      | ✅      |
| 🤖 Model Training          | ✅      |
| 🎯 Hyperparameter Tuning   | ✅      |
| 🎚️ Threshold Optimization | ✅      |
| 💰 Cost Analysis           | ✅      |
| 💾 Model Serialization     | ✅      |
| 🌐 Real-Time API           | 🔜     |
| 📊 Model Monitoring        | 🔜     |
| 🔄 Automated Retraining    | 🔜     |
| 🔍 Explainability Layer    | 🔜     |

---

# ⚠️ Limitations

This repository represents a **machine learning prototype / portfolio project**, not a production banking fraud-prevention system.

A real-world deployment would additionally require:

* 🔐 Secure infrastructure
* ⚡ Real-time inference
* 📊 Model monitoring
* 🔄 Data drift detection
* 🧠 Explainability
* 🔒 Privacy controls
* ⚖️ Regulatory compliance
* 🔁 Continuous retraining
* 🚨 Fraud investigation workflow

The financial values used in the business simulation are **project assumptions** and should not be interpreted as actual banking costs.

---

# 🧠 Technical Skills Demonstrated

```text
🐍 Python
📊 Data Analysis
🔎 Exploratory Data Analysis
🧹 Data Preprocessing
🧬 Feature Engineering
⚖️ Imbalanced Classification
🔵 SMOTE
🟠 Random Undersampling
📉 Logistic Regression
🌲 Random Forest
⚡ XGBoost
🎯 Hyperparameter Tuning
🔁 Cross-Validation
📈 Precision-Recall Analysis
🎚️ Threshold Optimization
💰 Cost-Benefit Analysis
💾 Model Serialization
🧩 ML Pipeline Design
```

# 🏆 Project Highlights

<div align="center">

### 🔥 284K+ Transactions

### ⚖️ Extreme Class Imbalance

### 🤖 3 ML Model Families

### 🔬 Hyperparameter Optimization

### 🎯 Threshold Engineering

### 💰 Financial Impact Simulation

### 💾 Reusable ML Pipeline

</div>

---

# 🌟 Why This Project Matters

Fraud detection demonstrates an important Data Science principle:

> ### **A high-performing ML model is not enough.**
>
> ### **The model must support a good business decision.**

This project combines:

```text
                 DATA
                  +
              MACHINE LEARNING
                  +
            IMBALANCE HANDLING
                  +
             MODEL TUNING
                  +
          THRESHOLD ENGINEERING
                  +
           BUSINESS ECONOMICS
                  ↓
        🛡️ FRAUD DETECTION SYSTEM
```

---

# 👨‍💻 Author

<div align="center">

## **Misari Dhorajiya**

### Data Science / AI-ML Enthusiast

🎓 Diploma in Information Technology
🏫 Government Polytechnic Gandhinagar

📚 Currently pursuing **AI & ML with Data Science**

<br>

<a href="https://github.com/dhorajiyamisri">
<img src="https://img.shields.io/badge/GitHub-dhorajiyamisri-111827?style=for-the-badge&logo=github&logoColor=white"/>
</a>

</div>

---

<div align="center">

## 💳 Built for Fraud Detection.

## 🧠 Designed for Machine Learning.

## 💰 Evaluated for Business Impact.

<br>

### ⭐ If you found this project useful, consider giving it a star!

<br>

**Made with 🐍 Python + 🤖 Machine Learning + 📊 Data Science**

</div>
