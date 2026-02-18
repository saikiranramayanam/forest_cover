Forest Cover Type Classification with LightGBM
1. Project overview
This project builds a high‑performance multi‑class classifier to predict forest cover type from cartographic features using the Forest Cover Type dataset. The focus is on:

End‑to‑end supervised learning workflow (EDA → feature engineering → model training → tuning → evaluation).

Using LightGBM as the primary model, with strong feature engineering and systematic hyperparameter tuning.

Handling class imbalance, analyzing per‑class performance, and comparing LightGBM with an alternative model (Random Forest).

2. Dataset description
The project uses the classic Forest Cover Type (Covertype) dataset, which contains cartographic variables describing forested areas and a target label indicating the dominant tree species.​ecies.​

Number of samples: ~581,000.​

Number of features: 54 input features + 1 target.​

Target variable:

Cover_Type – integer labels 1–7 representing different forest cover types.
*** Feature groups: ***

Continuous numeric features (10):

Elevation, Aspect, Slope,

Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology,

Horizontal_Distance_To_Roadways,

Hillshade_9am, Hillshade_Noon, Hillshade_3pm

Horizontal_Distance_To_Fire_Points.​

Wilderness area one‑hot features (4):

Wilderness_Area1–Wilderness_Area4.​
Soil type one‑hot features (40):

Soil_Type1–Soil_Type40.​
3. Objectives
The main goals are:​

Perform thorough EDA to understand feature distributions, correlations, and class imbalance.

Engineer at least five new features (including interaction and domain‑inspired features) to improve model performance.​

Build a baseline LightGBM model and then tune hyperparameters (e.g., num_leaves, learning_rate, feature_fraction, reg_alpha, reg_lambda) using k‑fold cross‑validation and early stopping.​

Handle class imbalance using class_weight or similar strategies.​

Evaluate the final model with macro/micro F1, precision, recall, confusion matrix, and feature importance (gain and split).​

Train at least one alternative classifier (Random Forest) and perform a comparative analysis of performance and behavior versus LightGBM.

4.Approach and methodology
4.1 Exploratory Data Analysis (EDA)
Key EDA steps:​

Checked shape, data types, missing values, and duplicates.

Visualized distributions of numeric features (histograms/KDE) and counts of Cover_Type, wilderness areas, and soil types.

Analyzed class imbalance in Cover_Type.

Computed correlation matrix between numeric features and target, and visualized selected feature vs target relationships (boxplots grouped by Cover_Type).

These insights guided feature engineering choices, especially around elevation, distances to hydrology/roads, and hillshade variables.

4.2 Feature engineering
Engineered features include:​

Hydrology_Dist_Sum = Horizontal_Distance_To_Hydrology + abs(Vertical_Distance_To_Hydrology)

Hydrology_Dist_Diff = Horizontal_Distance_To_Hydrology − abs(Vertical_Distance_To_Hydrology)

Road_Fire_Dist_Sum = Horizontal_Distance_To_Roadways + Horizontal_Distance_To_Fire_Points

Hillshade_Mean = average of Hillshade_9am, Hillshade_Noon, Hillshade_3pm

Hillshade_Range = max − min of the three hillshade values

Elevation_Slope_Interaction = Elevation × Slope

Soil_Type_Count = sum of all soil one‑hot columns (sanity check; should be 1)

Wilderness_Area_Encoded = single categorical feature derived from the wilderness one‑hot columns.

These features capture overall proximity to hydrology, human disturbance, illumination patterns, and terrain interactions, and enable LightGBM to exploit domain‑relevant interactions.

4.3 Data splitting and imbalance handling
Created stratified train/test split (80/20) to preserve class proportions using 0‑based labels for LightGBM (Cover_Type − 1).​

Used LightGBM’s native categorical support for Wilderness_Area_Encoded by setting it as category dtype and aligning categories between train and test.​

Addressed class imbalance via class_weight="balanced" in tuned LightGBM and Random Forest models.

4.4 Baseline LightGBM
Trained an initial LightGBM model with default‑ish parameters (multiclass objective, num_class=7) and early stopping.​

Evaluated baseline on the test set and via 5‑fold Stratified CV using macro F1 score.

4.5 Hyperparameter tuning
Implemented a systematic tuning process:​

Defined a CV helper using StratifiedKFold (k=5) and LightGBM with callbacks for early stopping and silent logging.

Designed a small parameter grid varying:

num_leaves

learning_rate

feature_fraction

reg_alpha (L1)

reg_lambda (L2)

class_weight (balanced)

For each config, ran 5‑fold CV and recorded macro F1 mean and std.

Selected the best configuration based on highest mean macro F1. This tuning improved CV macro F1 compared to the baseline configuration.

4.6 Final tuned LightGBM model
Re‑trained LightGBM on the training set using the best hyperparameters from tuning, with early stopping.​

Evaluated on the hold‑out test set, reporting:

Accuracy

Macro F1, micro F1

Macro precision, macro recall

Classification report (per‑class precision/recall/F1).​

Generated a confusion matrix to inspect per‑class performance and misclassification patterns.​

Extracted and plotted feature importance (both gain‑ and split‑based), highlighting which original and engineered features drive predictions (e.g., Elevation, hillshade features, hydrology distances, soil types).

4.7 Alternative model - Random Forest
Trained a RandomForestClassifier with n_estimators=200, class_weight="balanced", and default depth on the same train/test split and features.​

Evaluated using the same metrics as LightGBM (accuracy, macro/micro F1, macro precision/recall). This provides a strong tree‑based baseline from scikit‑learn for comparison.

4.8 Comparitive analysis - LightGBM vs Random Forest
Built a metrics DataFrame containing accuracy, macro/micro F1, macro precision, macro recall for both models.​

Visualized metrics with a bar plot and, optionally, side‑by‑side confusion matrices.​

Observed that tuned LightGBM generally achieves:

Slightly higher accuracy.

Higher macro F1 and macro recall, indicating better balanced performance across classes and better handling of minority classes.

5. Project Structure
. ├── data/ │ └── forest_cover.csv # Forest Cover Type dataset (not committed, or via instructions) ├── notebooks/ │ └── forest_covers_lightgbm.ipynb # Main project notebook ├── src/ # (Optional) helper modules, if any ├── requirements.txt # Python dependencies ├── README.md # This file └── environment.yml # (Optional) conda environment The primary notebook forest_covers_lightgbm.ipynb contains the full workflow and all visualizations

6. Environment setup
6.1 Using conda (recommended)
conda create -n forest-cover python=3.10 -y
conda activate forest-cover
pip install -r requirements.txt
6.2 Using venv
python -m  venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
6.3 Required Packages
Ensure the following packages are present in requirements.txt:​

lightgbm

scikit-learn

pandas

numpy

matplotlib

seaborn

jupyter (or notebook / jupyterlab)

7. How to run the notebook
1.Place the dataset

Download the Forest Cover Type dataset (CSV) and place it under data/ as:

data/forest_cover.csv
Ensure the path in the notebook matches this location.​

2.Activate environment

Activate the conda or venv environment created above.

3.Launch Jupyter

jupyter notebook
# or
jupyter lab
4.Open the notebook

Navigate to notebooks/forest_covers_lightgbm.ipynb.

Run all cells in order (Kernel -> Restart & Run All).​

5.Outputs

EDA plots (distributions, correlations).

Engineered feature summaries.

Baseline and tuned LightGBM training logs.

Evaluation metrics, confusion matrices, feature importance plots.

Random Forest metrics and comparative plots.

8. Results
final test metrics :

model	accuracy	macro_f1	micro_f1	macro_precision	macro_recall
LightGBM_tuned	0.9595	0.9421	0.9595	0.9505	0.9342
RandomForest	0.9578	0.9283	0.9578	0.9468	0.9121
Tuned LightGBM outperforms Random Forest on most metrics, especially macro F1 and macro recall, indicating better per‑class performance and minority class handling.​

Tuning improves LightGBM relative to its baseline configuration (higher macro F1 and improved validation performance).

9. Future work
Potential extensions:​

Use more advanced hyperparameter search (e.g., Optuna or Hyperopt) with larger search spaces.

Experiment with other gradient boosting frameworks (e.g., XGBoost, CatBoost) for comparison.

Engineer additional domain‑specific features or learn embeddings for soil and wilderness categories.

Apply calibration and thresholding strategies to optimize specific metrics (e.g., macro recall for rare classes).