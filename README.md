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
| __SlogP_VSA10__ | __5/5__ |
| __SlogP_VSA1__ | __5/5__ |
| __SMR_VSA10__ | __5/5__ |
| EState_VSA1 | 4/5 |
| fr_NH2 | 3/5 |
| SMR_VSA3 | 3/5 |
| PEOE_VSA8 | 3/5 |
| NumAliphaticRings | 3/5 |

The first three descriptors appear in the top 10 for every imbalance strategy. This indicates **consistency across these five fitted models**, not demonstrated robustness across independent training samples or proof of a causal biological relationship.

### SHAP dependence plots

The notebook generates __15 dependence plots__: three recurring descriptors (`SlogP_VSA10`, `SlogP_VSA1`, `SMR_VSA10`) across five imbalance strategies.

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

## Summary

This study investigated the prediction of **drug-induced autoimmunity (DIA)** using machine learning, with particular emphasis on feature selection, class imbalance, predictive performance, and model interpretability. Using molecular descriptors derived from chemical structures and the dataset associated with Huang et al. (2025), the research explored how different modeling decisions influence both the identification of potentially autoimmunity-inducing drugs and the molecular features used to explain those predictions.

### 1. Machine Learning Models and Feature Selection

Three machine learning classifiers were investigated:

- Logistic Regression (LR)
- Random Forest (RF)
- Support Vector Machine (SVM)

These models were evaluated alongside three feature selection approaches: **Mutual Information (MI), Recursive Feature Elimination with Cross-Validation (RFECV), and Genetic Algorithm (GA)**.

The original dataset contained 196 RDKit molecular descriptors. Following preprocessing, including low-variance feature removal and correlation filtering, 140 descriptors remained for further analysis.

The feature selection experiments examined whether reducing the number of molecular descriptors could preserve predictive information while improving model efficiency and interpretability. They also demonstrated the importance of evaluating feature selection jointly with the classifier rather than assuming that one feature subset will perform equally well across different algorithms.

Random Forest was subsequently used for the detailed comparison of class-imbalance handling strategies and SHAP-based interpretation.

### 2. Class Imbalance and External Test Performance

Because DIA-positive drugs represented a minority of the dataset, class imbalance was an important methodological challenge. Five strategies were compared using Random Forest:

1. Baseline (no imbalance correction)
2. Synthetic Minority Oversampling Technique (SMOTE)
3. Class Weight Adjustment
4. Random Oversampling (ROS)
5. Random Undersampling (RUS)

The models were evaluated on an independent external test set containing **120 drugs: 90 DIA-negative and 30 DIA-positive**.

| Strategy | MCC | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.510 | 0.833 | 0.857 | 0.400 | 0.545 | 0.881 | 0.740 |
| **SMOTE** | **0.542** | **0.842** | **0.923** | 0.400 | 0.558 | 0.893 | **0.794** |
| Class Weight | 0.484 | 0.825 | 0.909 | 0.333 | 0.488 | **0.901** | 0.751 |
| ROS | 0.450 | 0.817 | 0.786 | 0.367 | 0.500 | 0.885 | 0.726 |
| RUS | 0.460 | 0.742 | 0.490 | **0.800** | **0.608** | 0.850 | 0.719 |

**SMOTE achieved the highest Matthews Correlation Coefficient (MCC = 0.542)**, the highest precision (0.923), and the highest PR-AUC (0.794), indicating the strongest overall balance among the evaluated strategies according to the primary MCC criterion.

However, SMOTE correctly identified only **12 of the 30 DIA-positive drugs**, corresponding to a recall of 40%. Its high precision was accompanied by a substantial number of missed positive cases.

In contrast, **Random Undersampling achieved 80% recall**, correctly identifying 24 of the 30 DIA-positive drugs. This improvement came at the cost of 25 false-positive predictions, compared with only one false positive for SMOTE.

This demonstrates a central finding of the study: **the preferred imbalance-handling strategy depends on the relative importance of false-negative and false-positive predictions**.

In early-stage drug safety screening, where failing to flag a potentially harmful compound may be particularly concerning, a higher-recall model such as RUS may be useful as a preliminary screening tool. However, its higher false-positive rate could increase the number of compounds requiring additional investigation.

SMOTE may be more appropriate when a smaller number of high-confidence positive predictions is preferred.

Neither strategy should be considered sufficient for clinical or regulatory decisions without additional validation.

### 3. SHAP Explainability and Molecular Descriptor Importance

SHapley Additive exPlanations (SHAP) were used to investigate how molecular descriptors contributed to Random Forest predictions across the five imbalance-handling strategies.

The analysis included global feature importance, beeswarm plots, dependence plots, individual prediction explanations, and comparisons of the most influential descriptors across models.

Several molecular descriptors repeatedly appeared among the most influential features:

- **SlogP_VSA10**
- **SlogP_VSA1**
- **SMR_VSA10**
- **EState_VSA1**

SlogP_VSA10, SlogP_VSA1, and SMR_VSA10 appeared in the top ten SHAP features across all five imbalance strategies, while EState_VSA1 appeared in four.

These findings indicate that molecular properties represented by these descriptors consistently contributed to the trained models' predictions, despite differences in the class-imbalance strategy.

However, the average top-ten feature overlap across strategies was approximately **53%**, suggesting that a substantial portion of the model explanations changed when the imbalance-handling method changed.

Therefore, the choice of imbalance strategy affects not only classification performance but also **which molecular properties appear most important to the model**.

This is particularly relevant in interpretable machine learning for drug safety, where explanations may be used to generate scientific hypotheses.

### 4. Chemical Interpretation of the Important Descriptors

The consistently important molecular descriptors have meaningful connections to physicochemical properties of drug molecules.

**SlogP_VSA1 and SlogP_VSA10** are molecular surface-area descriptors grouped according to atom contributions to estimated lipophilicity (SlogP). They encode aspects of the distribution of hydrophobic and hydrophilic characteristics across a molecule.

Lipophilicity is relevant to drug absorption, membrane partitioning, tissue distribution, and interactions with biological environments. Differences in these properties can influence drug exposure and the opportunities for interactions with cellular components.

**SMR_VSA10** represents molecular surface-area information grouped according to atomic contributions to molar refractivity. Molar refractivity is associated with molecular polarizability and volume-related properties, which can be relevant to noncovalent molecular interactions.

**EState_VSA1** combines molecular surface-area information with electrotopological state characteristics, providing a representation of aspects of the electronic and topological environments of atoms within a molecule.

The recurring importance of these descriptors suggests that **surface-area-related physicochemical properties, lipophilicity-associated features, and electronic/topological molecular characteristics contain predictive information for DIA classification in this dataset**.

These results are associations learned by the models. They do not establish that any individual descriptor directly causes drug-induced autoimmunity.

### 5. Biological Relevance to Drug-Induced Autoimmunity

Drug-induced autoimmunity is a complex biological phenomenon involving interactions among drug exposure, metabolism, immune regulation, and individual susceptibility.

Potential biological mechanisms include drug-related changes in self-antigen recognition, formation of reactive metabolites or drug-protein adducts, altered immune signaling, and the activation of autoreactive immune responses.

The molecular descriptors highlighted by SHAP may be indirectly relevant to these processes through their relationships with drug distribution, chemical reactivity, molecular interactions, and biological exposure.

For example, lipophilicity-associated properties may influence how a compound distributes across tissues and cellular membranes, potentially affecting the environments in which drug-related immune interactions occur.

Likewise, electronic and molecular surface-area characteristics may be relevant to how drug molecules interact with proteins and other biological structures.

However, **the present study did not directly measure immune activation, protein binding, metabolite formation, or autoimmune pathways**. Consequently, the SHAP findings should be interpreted as potential starting points for chemical and biological investigation rather than evidence of a specific immunological mechanism.

Experimental toxicology studies, mechanistic assays, and independent chemical validation would be necessary to determine whether these molecular characteristics have a causal role in DIA.

### 6. Stability of Model Explanations

A further observation was that SHAP explanations were only partially consistent across the five imbalance-handling strategies.

Although several molecular descriptors remained important across all strategies, the approximately 53% average overlap among the top ten descriptors indicates that explanation rankings were sensitive to the training procedure.

Individual prediction explanations also differed across strategies. For one examined drug, predicted DIA-positive probabilities ranged from approximately 0.05 to 0.13, depending on the model.

This suggests that **changes in class-imbalance handling can influence both predicted risk scores and the apparent molecular drivers of those predictions**.

Accordingly, model interpretability should not be evaluated solely by generating SHAP plots for a single model. Comparing explanations across reasonable modeling choices provides additional information about their robustness.

### 7. Limitations and Future Research

Several limitations should be considered when interpreting these results.

First, the dataset was relatively small, particularly the external DIA-positive test group, which contained only 30 compounds. As a result, performance metrics such as recall and MCC may be sensitive to a small number of classification changes.

Second, the models relied on calculated molecular descriptors rather than direct measurements of biological pathways, immune responses, or patient-specific susceptibility.

Third, some preprocessing operations may have occurred before cross-validation rather than being refitted independently within every training fold. If so, this could introduce information leakage into cross-validation estimates and should be addressed in future work.

Fourth, SHAP measures contributions to model predictions rather than chemical causality or biological mechanisms.

Future studies could investigate larger and more chemically diverse datasets, scaffold-based validation, additional external datasets, probability calibration, and experimental assessment of the molecular hypotheses suggested by SHAP.

### Overall Conclusion

**This research demonstrates that class-imbalance handling is a critical factor in machine learning-based prediction of drug-induced autoimmunity, affecting both predictive performance and model interpretability.**

SMOTE provided the highest external-test MCC and PR-AUC, whereas Random Undersampling substantially increased the detection of DIA-positive drugs at the expense of more false positives. These findings highlight the importance of choosing evaluation metrics and decision thresholds according to the intended drug safety screening objective.

The SHAP analysis further identified recurring molecular descriptors related to lipophilicity-associated surface area, molar refractivity, and electrotopological properties. Their repeated importance suggests that physicochemical characteristics of drug molecules contain useful predictive signals for DIA classification.

At the same time, the variation in SHAP feature rankings across imbalance strategies emphasizes that machine learning explanations depend partly on modeling choices and should not be interpreted as definitive biological mechanisms.

**Overall, the study connects computational chemistry, machine learning, and drug safety research by showing how molecular descriptors can support predictive screening and generate hypotheses for further biological investigation.** The findings provide a foundation for future work combining robust external validation, interpretable modeling, and experimental toxicology to improve the identification and understanding of potentially autoimmunity-inducing drugs.

## References

1. Huang, L., Liu, P., & Huang, X. (2025). *InterDIA: Interpretable Prediction of Drug-induced Autoimmunity through Ensemble Machine Learning Approaches.* Toxicology, Article 154064. https://github.com/Huangxiaojie2024/InterDIA

2. Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems, 30. https://arxiv.org/abs/1705.07874
