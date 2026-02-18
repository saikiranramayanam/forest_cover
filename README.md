# 🌲 Forest Cover Type Classification with LightGBM

## 📌 Project Overview

This project builds a high-performance multi-class classifier to predict forest cover type from cartographic features using the **Forest Cover Type (Covertype) dataset**.

The focus is on:

- End-to-end supervised learning workflow (EDA → Feature Engineering → Model Training → Tuning → Evaluation)
- Using **LightGBM** as the primary model
- Systematic hyperparameter tuning with cross-validation
- Handling class imbalance
- Per-class performance analysis
- Comparison with **Random Forest**

---

## 📊 Dataset Description

The project uses the classic **Forest Cover Type dataset**, containing cartographic variables describing forested areas and a target label indicating the dominant tree species.

- **Samples:** ~581,000  
- **Features:** 54 input features + 1 target  
- **Target Variable:**  
  `Cover_Type` (integer labels 1–7 representing forest cover types)

---

### 🔹 Feature Groups

#### 1️⃣ Continuous Numeric Features (10)

- Elevation
- Aspect
- Slope
- Horizontal_Distance_To_Hydrology
- Vertical_Distance_To_Hydrology
- Horizontal_Distance_To_Roadways
- Hillshade_9am
- Hillshade_Noon
- Hillshade_3pm
- Horizontal_Distance_To_Fire_Points

#### 2️⃣ Wilderness Area (One-Hot, 4)

- Wilderness_Area1 – Wilderness_Area4

#### 3️⃣ Soil Type (One-Hot, 40)

- Soil_Type1 – Soil_Type40

---

## 🎯 Objectives

- Perform thorough EDA
- Engineer at least five meaningful features
- Build baseline LightGBM
- Tune hyperparameters using Stratified K-Fold CV
- Handle class imbalance
- Evaluate using macro/micro F1, precision, recall
- Compare against Random Forest

---

# 🧪 Methodology

## 1️⃣ Exploratory Data Analysis (EDA)

- Checked shape, data types, missing values, duplicates
- Visualized feature distributions
- Analyzed class imbalance
- Correlation analysis
- Feature vs target boxplots

**Key Insight:** Elevation, hydrology distances, and hillshade variables strongly influence forest cover type.

---

## 2️⃣ Feature Engineering

Engineered features:

```python
Hydrology_Dist_Sum
Hydrology_Dist_Diff
Road_Fire_Dist_Sum
Hillshade_Mean
Hillshade_Range
Elevation_Slope_Interaction
Soil_Type_Count
Wilderness_Area_Encoded
These capture:

Terrain interaction

Illumination patterns

Hydrological proximity

Human disturbance impact

3️⃣ Data Splitting & Imbalance Handling
Stratified 80/20 train-test split

Converted labels to 0-based for LightGBM

Used native categorical handling for wilderness area

Applied class_weight="balanced"

4️⃣ Baseline LightGBM
Multiclass objective

Early stopping

5-fold Stratified CV

Evaluated using macro F1

5️⃣ Hyperparameter Tuning
Parameters tuned:

num_leaves

learning_rate

feature_fraction

reg_alpha

reg_lambda

class_weight

Used:

StratifiedKFold (k=5)

Early stopping

Macro F1 as optimization metric

Result: Improved CV macro F1 over baseline.

6️⃣ Final Tuned LightGBM Evaluation
Reported:

Accuracy

Macro F1

Micro F1

Macro Precision

Macro Recall

Classification Report

Confusion Matrix

Feature Importance (Gain + Split)

7️⃣ Alternative Model — Random Forest
n_estimators=200

class_weight="balanced"

Evaluated using same metrics for fair comparison.

📈 Comparative Analysis
Model	Accuracy	Macro F1	Micro F1	Macro Precision	Macro Recall
LightGBM (Tuned)	0.9595	0.9421	0.9595	0.9505	0.9342
Random Forest	0.9578	0.9283	0.9578	0.9468	0.9121
🔎 Observations
LightGBM achieves slightly higher accuracy

Higher macro F1 → better balanced performance

Higher macro recall → better minority class handling

Tuning improves validation stability

📁 Project Structure
.
├── data/
│   └── forest_cover.csv
├── notebooks/
│   └── forest_covers_lightgbm.ipynb
├── src/
├── requirements.txt
├── README.md
└── environment.yml
⚙️ Environment Setup
Option 1: Conda (Recommended)
conda create -n forest-cover python=3.10 -y
conda activate forest-cover
pip install -r requirements.txt
Option 2: venv
python -m venv .venv
Windows
.venv\Scripts\activate
Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
📦 Required Packages
lightgbm

scikit-learn

pandas

numpy

matplotlib

seaborn

jupyter

▶️ How to Run
Place dataset:

data/forest_cover.csv
Activate environment

Launch Jupyter

jupyter notebook
or

jupyter lab
Open:

notebooks/forest_covers_lightgbm.ipynb
Run all cells

🏆 Final Results
Model	Accuracy	Macro F1	Micro F1	Macro Precision	Macro Recall
LightGBM (Tuned)	0.9595	0.9421	0.9595	0.9505	0.9342
Random Forest	0.9578	0.9283	0.9578	0.9468	0.9121
Conclusion: Tuned LightGBM outperforms Random Forest, particularly on macro F1 and macro recall.

🚀 Future Work
Use Optuna/Hyperopt for advanced tuning

Compare with XGBoost and CatBoost

Additional domain-driven feature engineering

Model calibration for rare-class optimization

👤 Author
Sai Kiran Ramayanam
