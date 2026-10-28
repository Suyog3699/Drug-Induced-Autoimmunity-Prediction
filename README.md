# Drug-Induced Autoimmunity Prediction Using Machine Learning

**An exploratory study of feature selection, class-imbalance strategies, and SHAP explainability using RDKit molecular descriptors.**

## Overview

Drug-induced autoimmunity (DIA) is the focus of this binary classification project. The aim is to investigate how molecular descriptors can be used to distinguish DIA-positive from DIA-negative drugs, compare feature-selection approaches and classifiers, and examine the effect of class-imbalance treatments on predictive performance and model explanations.

This repository documents an **exploratory machine learning research project**. The reported results are those obtained in the accompanying Jupyter notebook; they are not claims of clinical validity or prospective drug-safety performance.

## Research questions

1. How do Mutual Information (MI), Recursive Feature Elimination with Cross-Validation (RFECV), and Genetic Algorithm (GA) feature selection affect classifier performance?
2. How do Logistic Regression, Random Forest, and Support Vector Machine compare on the available molecular-descriptor data?
3. How do Baseline, SMOTE, Class Weight, Random Oversampling (ROS), and Random Undersampling (RUS) affect Random Forest classification of DIA-positive drugs?
4. How consistent are SHAP molecular-descriptor importance rankings and individual explanations across imbalance strategies?

## Dataset

The notebook loads separate training and external test CSV files containing RDKit molecular descriptors and binary labels (`0` = DIA-negative, `1` = DIA-positive).

| Split | DIA-negative | DIA-positive | Total |
|---|---:|---:|---:|
| Training | 359 | 118 | 477 |
| External test | 90 | 30 | 120 |
| **Total** | **449** | **148** | **597** |

- Starting feature set: **196 molecular descriptors**.
- The notebook reports **no missing feature values** in either split.
- After standardization, variance filtering, and correlation filtering, **140 descriptors** remain.
- The later Random Forest imbalance comparison uses **93 features selected by Mutual Information** on the training set.

**Data provenance:** Add the original dataset publication/repository URL, citation, license, and download instructions here before making a public release. The HTML notebook alone does not establish redistribution rights. Do not upload the CSV files unless permitted.

## Methodology

### 1. Preprocessing

The notebook implements:

- Standardization with `StandardScaler`.
- Low-variance feature removal with `VarianceThreshold`.
- Correlation-based feature filtering using a high-correlation threshold (absolute correlation above 0.90).
- Application of the selected training-derived feature columns to the external test data.

### 2. Feature-selection comparison

Three approaches are compared:

- **Mutual Information (MI)** — selects descriptors associated with the binary target.
- **RFECV** — recursively eliminates features using cross-validation.
- **Genetic Algorithm (GA)** — searches feature subsets using a genetic selection procedure.

The notebook evaluates the feature-selection methods with **5-fold stratified cross-validation** using Logistic Regression, Random Forest, and SVM. The metrics include MCC, accuracy, precision, recall, specificity, F1, ROC-AUC, and PR-AUC.

#### Logistic Regression — cross-validation mean

| Selection | MCC | Accuracy | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MI | **0.3552** | **0.7758** | **0.4149** | **0.4786** | **0.7328** | **0.5013** |
| RFECV | 0.2637 | 0.7591 | 0.3333 | 0.3909 | 0.6779 | 0.4744 |
| GA | 0.2630 | 0.7529 | 0.3580 | 0.4061 | 0.6890 | 0.4477 |

#### Random Forest — cross-validation mean

| Selection | MCC | Accuracy | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MI | **0.4921** | **0.8281** | 0.3304 | **0.4856** | **0.8324** | **0.7279** |
| RFECV | 0.4638 | 0.8218 | 0.3308 | 0.4788 | 0.8316 | 0.7092 |
| GA | 0.4423 | 0.8113 | **0.3558** | 0.4845 | 0.8064 | 0.6843 |

#### Support Vector Machine — cross-validation mean

| Selection | MCC | Accuracy | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| MI | 0.3186 | 0.7633 | 0.3989 | 0.4572 | **0.7263** | **0.5070** |
| RFECV | 0.2793 | 0.7613 | 0.3841 | 0.4194 | 0.6838 | 0.4882 |
| GA | **0.3383** | **0.7675** | **0.4246** | **0.4743** | 0.6720 | 0.4754 |

**Interpretation:** Random Forest with MI has the highest cross-validation MCC among the classifier/feature-selection combinations shown (0.4921). MI is also the highest-MCC feature-selection approach for Logistic Regression, while GA is the highest-MCC approach for SVM. These are *exploratory* CV estimates; see the preprocessing limitation below.

### 3. Random Forest class-imbalance comparison

For the subsequent imbalance experiment, the notebook uses a Random Forest classifier and a common MI-selected descriptor set, comparing:

1. **Baseline** — no explicit imbalance treatment.
2. **SMOTE** — synthetic minority oversampling.
3. **Class Weight** — increased weight for the DIA-positive class.
4. **ROS** — random oversampling.
5. **RUS** — random undersampling.

The imbalance treatments are applied to training data, not the external test set. The external test set contains the original 90 negative and 30 positive drugs.

## External test results

All values below are from the notebook's external-test evaluation at the model's default classification threshold.

| Strategy | MCC | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.5096 | 0.8333 | 0.8571 | 0.4000 | 0.9778 | 0.5455 | 0.8806 | 0.7403 |
| **SMOTE** | **0.5418** | **0.8417** | **0.9231** | 0.4000 | **0.9889** | 0.5581 | 0.8933 | **0.7936** |
| Class Weight | 0.4835 | 0.8250 | 0.9091 | 0.3333 | **0.9889** | 0.4878 | **0.9015** | 0.7515 |
| ROS | 0.4496 | 0.8167 | 0.7857 | 0.3667 | 0.9667 | 0.5000 | 0.8850 | 0.7264 |
| RUS | 0.4601 | 0.7417 | 0.4898 | **0.8000** | 0.7222 | **0.6076** | 0.8504 | 0.7186 |

### Confusion matrices

Confusion-matrix counts are given as true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP).

| Strategy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Baseline | 88 | 2 | 18 | 12 |
| SMOTE | 89 | 1 | 18 | 12 |
| Class Weight | 89 | 1 | 20 | 10 |
| ROS | 87 | 3 | 19 | 11 |
| RUS | 65 | 25 | 6 | 24 |

**Main trade-off:** SMOTE has the highest external-test MCC (0.5418) and PR-AUC (0.7936) among these five models, while RUS achieves the highest recall (0.80), identifying 24 of the 30 DIA-positive drugs. However, RUS also generates 25 false positives, compared with one under SMOTE. No single strategy is best on every metric.

## SHAP explainability analysis

The notebook uses `shap.TreeExplainer` to explain the **DIA-positive class (class 1)** of the five Random Forest models.

### Global explanations

For each strategy, the analysis computes mean absolute SHAP values and generates:

- Top-15 SHAP feature-importance bar plots.
- Top-15 SHAP beeswarm plots.
- A combined feature-importance comparison table.

### Ranking agreement across strategies

Spearman correlations of feature-importance rankings are:

| | Baseline | SMOTE | Class Weight | ROS | RUS |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.000 | 0.822 | 0.845 | 0.871 | 0.852 |
| SMOTE | 0.822 | 1.000 | 0.817 | 0.838 | 0.799 |
| Class Weight | 0.845 | 0.817 | 1.000 | **0.951** | 0.886 |
| ROS | 0.871 | 0.838 | **0.951** | 1.000 | 0.886 |
| RUS | 0.852 | 0.799 | 0.886 | 0.886 | 1.000 |

All off-diagonal correlations are positive and relatively high (**0.799–0.951**). The highest ranking agreement is between Class Weight and ROS; the lowest is between SMOTE and RUS.

### Top-10 feature overlap

The number of shared descriptors in each pair of top-10 lists is:

| | Baseline | SMOTE | Class Weight | ROS | RUS |
|---|---:|---:|---:|---:|---:|
| Baseline | 10 | 5 | 5 | 7 | 5 |
| SMOTE | 5 | 10 | 4 | 5 | 5 |
| Class Weight | 5 | 4 | 10 | 7 | 5 |
| ROS | 7 | 5 | 7 | 10 | 5 |
| RUS | 5 | 5 | 5 | 5 | 10 |

Across the 10 unique model pairs, top-10 overlap ranges from **4/10 to 7/10**, averaging **5.3/10 (53%)**. Thus, broad feature rankings agree more than the membership of the top-10 lists.

### Consistently important descriptors

| Descriptor | Number of strategies placing it in the top 10 |
|---|---:|
| **SlogP_VSA10** | **5/5** |
| **SlogP_VSA1** | **5/5** |
| **SMR_VSA10** | **5/5** |
| EState_VSA1 | 4/5 |
| fr_NH2 | 3/5 |
| SMR_VSA3 | 3/5 |
| PEOE_VSA8 | 3/5 |
| NumAliphaticRings | 3/5 |

The first three descriptors appear in the top 10 for every imbalance strategy. This indicates **consistency across these five fitted models**, not demonstrated robustness across independent training samples or proof of a causal biological relationship.

### SHAP dependence plots

The notebook generates **15 dependence plots**: three recurring descriptors (`SlogP_VSA10`, `SlogP_VSA1`, `SMR_VSA10`) across five imbalance strategies.

Visual inspection suggests:

- `SlogP_VSA10`: lower descriptor values tend to contribute negatively to the DIA-positive prediction, with higher values often contributing positively.
- `SlogP_VSA1`: a nonlinear relationship, with the sign and magnitude of contributions varying over the descriptor range.
- `SMR_VSA10`: a nonlinear relationship; low and some high values can have negative contributions, while intermediate values can have positive contributions.

These are **model-explanation patterns**, not evidence of causation or experimentally validated chemical mechanisms.

### Local explanations: waterfall plots

The notebook also generates waterfall plots for the same external-test drug under all five models. One verified Baseline example has:

- SHAP base value: `0.2487002096`
- Sum of SHAP contributions: `-0.1887002096`
- Final DIA-positive prediction: `0.0600`
- Direct Random Forest predicted probability: `0.0600`

The SHAP decomposition matches the model's predicted probability, showing how descriptor contributions move the prediction from the expected output to the individual prediction. The notebook compares local explanations for this same drug across the five strategies.

## Conclusions

1. In the reported exploratory cross-validation experiments, **Random Forest with Mutual Information feature selection** achieved the highest MCC among the classifier/selection combinations examined.
2. On the external test set, **SMOTE** produced the highest MCC and PR-AUC, while **RUS** substantially increased recall but also increased false positives.
3. SHAP feature-importance rankings were broadly similar across the five fitted Random Forest models (Spearman ρ = 0.799–0.951), although top-10 feature overlap was only 40–70%.
4. `SlogP_VSA10`, `SlogP_VSA1`, and `SMR_VSA10` were among the top 10 SHAP descriptors in all five strategies.
5. The results illustrate a trade-off between detecting DIA-positive drugs, limiting false alarms, and the consistency of model explanations.

## Limitations and future work

**Important methodological limitation:** The current notebook performs data-dependent preprocessing (including standardization and correlation filtering) before the classifier-comparison cross-validation. These transformations should instead be fitted **within each training fold**, together with feature selection and any resampling. Consequently, the cross-validation results reported above should be treated as preliminary and may be optimistic.

Other limitations:

- Only **30 DIA-positive drugs** are present in the external test set; uncertainty intervals are not yet reported.
- The project compares the five strategies primarily at the default **0.50 classification threshold**; threshold optimization should use training/validation data, not external-test labels.
- SHAP ranking agreement is based on **one fitted model per strategy**, not repeated seeds or bootstrap stability estimates.
- Molecular descriptors and SHAP values explain model behavior, **not causality or clinical risk**.
- External reproducibility depends on obtaining the original dataset, installing compatible package versions, and replacing machine-specific file paths.

Potential extensions include leakage-safe nested or repeated cross-validation, uncertainty estimates, threshold analysis, additional independent validation, and repeated-run SHAP stability analysis.

## Repository contents and how to run

Recommended layout:

```text
drug-induced-autoimmunity-prediction/
├── README.md
├── requirements.txt
├── notebooks/
│   └── DIA_Research.ipynb
├── data/
│   └── README.md                 # Dataset source and download instructions
├── results/                     # Selected tables and confusion matrices
└── figures/                     # Selected SHAP figures
```

1. Obtain the dataset from its original authorized source.
2. Place the training and external test CSVs in `data/` (or adjust the notebook's relative paths).
3. Install the dependencies documented in `requirements.txt` and open `notebooks/DIA_Research.ipynb`.
4. Run cells in order. The original notebook contains absolute local paths that must be updated before it will run on another computer.

Core Python packages used include `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `shap`, `matplotlib`, `joblib`, and a package providing `genetic_selection.GeneticSelectionCV`. Record the exact versions from the environment used for the final run to improve reproducibility.

## Status

**Exploratory undergraduate research project / portfolio implementation.** Results are presented transparently, including methodological limitations. This repository is not a validated clinical decision-support tool.
