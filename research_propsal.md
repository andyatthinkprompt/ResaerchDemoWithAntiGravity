# Research Proposal & Execution Spec
## Project: Bank Marketing Customer Classification (Traditional ML)

---

## 1. Overview

**Goal**  
Build a reproducible machine learning pipeline to classify potential customers for bank marketing campaigns and evaluate the impact of:
- Feature engineering
- Class imbalance handling
- Model selection (traditional ML)

---

## 2. Research Objective

Perform a **comparative analysis of traditional ML algorithms** for predicting customer subscription, focusing on:

- Performance differences across models
- Impact of preprocessing strategies
- Effectiveness of imbalance handling techniques
- Model interpretability

---

## 3. Research Questions

1. Which traditional ML model performs best?
2. How does feature engineering affect performance?
3. What is the impact of class imbalance techniques?
4. Which features are most influential?

---

## 4. Dataset

### Source
- Kaggle: Bank Marketing Dataset (UCI-based)

### Requirements
- Must be downloadable programmatically
- Store raw dataset in `/data/raw/`
- Store processed dataset in `/data/processed/`

### Notes
- Target column: `y` (yes/no)
- Expect class imbalance (~10–15% positive)
- Identify and optionally drop leakage feature (`duration`)

---

## 5. System Architecture


data/
raw/
processed/

src/
data/
features/
models/
evaluation/
visualization/

outputs/
tables/
charts/
logs/

notebooks/


---

## 6. Pipeline Specification

### Step 1: Data Collection

#### Tasks
- Download dataset from Kaggle
- Save locally
- Load into pandas DataFrame

#### Output
- `data/raw/bank.csv`

---

### Step 2: Data Preprocessing

#### Tasks
- Handle missing values (if any)
- Convert target:
  - yes → 1
  - no → 0
- Identify categorical vs numerical features

#### Encoding Strategies
- One-hot encoding
- Label encoding (for comparison)

#### Output
- Clean DataFrame
- Feature matrix `X`
- Target vector `y`

---

### Step 3: Feature Engineering

#### Tasks
- Remove leakage feature (`duration`) → optional experiment
- Feature selection:
  - Chi-square (categorical)
  - Correlation filtering (numerical)
- Store feature sets:
  - Full features
  - Reduced features

---

### Step 4: Train/Test Split

#### Requirements
- Stratified split
- Test size: 20%
- Fixed random seed (42)

---

### Step 5: Class Imbalance Handling

#### Techniques
- None (baseline)
- Class weights
- SMOTE

#### Output
- Multiple training sets:
  - Original
  - Weighted
  - SMOTE-resampled

---

### Step 6: Model Training

#### Models to Implement
- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- k-NN
- Naive Bayes

#### Requirements
- Use consistent interface
- Store trained models
- Track hyperparameters (default first, optional tuning later)

---

### Step 7: Validation Strategy

#### Method
- Stratified k-fold cross-validation (k=5)

#### Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

### Step 8: Evaluation

#### Tasks
- Evaluate all models on test set
- Compare across:
  - Models
  - Encoding methods
  - Imbalance techniques

#### Output Tables
- `outputs/tables/model_comparison.csv`

Columns:

Model | Encoding | Imbalance | Accuracy | Precision | Recall | F1 | ROC-AUC


---

### Step 9: Visualization

#### Required Charts
1. ROC curves (all models)
2. Model comparison bar chart (F1 / ROC-AUC)
3. Feature importance (Random Forest)
4. Confusion matrix (best model)

#### Output
- `outputs/charts/*.png`

---

### Step 10: Interpretability

#### Tasks
- Extract:
  - Logistic Regression coefficients
  - Random Forest feature importance
- Rank top features
- Provide explanation of key drivers

---

## 7. Experiment Matrix

| Dimension            | Variants                      |
|---------------------|------------------------------|
| Encoding            | One-hot, Label               |
| Imbalance Handling  | None, Class Weight, SMOTE    |
| Models              | 6 models                     |

Total runs ≈ 36 experiments

---

## 8. Deliverables

### Code
- Modular Python pipeline
- Reproducible (fixed seeds, deterministic)

### Data Outputs
- Clean dataset
- Feature sets

### Tables
- Model comparison table
- Feature importance table

### Charts
- ROC curves
- Performance comparison
- Feature importance

### Report Sections
- Methodology
- Results
- Discussion
- Limitations

---

## 9. Success Criteria

- All models trained and evaluated
- Comparison table completed
- At least 3 meaningful insights:
  - Best model
  - Impact of imbalance handling
  - Key predictive features

---

## 10. Constraints

- Use only traditional ML (no deep learning)
- Use Kaggle dataset
- Keep pipeline interpretable
- Avoid data leakage

---

## 11. Optional Extensions

- Hyperparameter tuning (GridSearchCV)
- SHAP values for interpretability
- Precision@K (business metric)
- Remove vs include `duration` comparison

---

## 12. Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- imbalanced-learn
- matplotlib / seaborn

---

## 13. Execution Notes (for Agent)

- Keep code modular (no monolithic notebook)
- Log every experiment configuration
- Save intermediate outputs
- Ensure reproducibility
- Avoid hardcoding paths

---

## 14. Final Output Format

Agent should produce:


outputs/
tables/
model_comparison.csv
charts/
roc_curve.png
model_performance.png
feature_importance.png


---

## 15. Key Insight Targets

Agent should explicitly answer:

1. Which model performs best and why?
2. Does SMOTE improve recall significantly?
3. Does feature engineering matter more than model choice?
4. Which features drive predictions?

---

END OF SPEC