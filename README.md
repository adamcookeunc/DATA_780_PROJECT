<img width="1200" height="686" alt="image" src="https://github.com/user-attachments/assets/80b034a3-3b88-4726-a5e5-4930c2539aa6" />



# Model Selection for Customer Churn Prediction in the Telecommunications Industry

 

## Overview

 

This project presents a comprehensive machine learning analysis for predicting customer churn in the telecommunications industry. It evaluates multiple classification models, sampling and scaling techniques, and applies hyperparameter tuning to optimize performance. The final models are tested across multiple datasets, including those from different industries, to assess generalizability.

 

---

 

## Project Repository Structure

 

- `DATA_780_PROJECT_NOTEBOOK_07272025.html`: A rendered version of the project notebook with data preprocessing, EDA, model training, evaluation, SHAP analysis, and performance metrics comparisons.

 

- `DATA_780_PROJECT_NOTEBOOK_07272025.ipynb`: The actual Jupyter notebook used to perform the analysis for the proect with code that can be edited and executed.

 

- `DATA_780_Powerpoint.pdf`: A slideshow presentation summarizing the methodology, results, and key findings.

 

---

 

## Objectives

 

- Compare combinations of sampling, scaling, and classification models to identify the optimal model configuration combinations for each model type and

- Apply the optimized model to new datasets to evaluate generalizability.

- Use SHAP (Shapley Additive Explanations) to interpret model predictions and identify key churn indicators.

 

---

 

## Methods

 

### Models Compared

- Logistic Regression

- Decision Tree

- Random Forest

- Support Vector Machine (SVM)

- Gaussian Naive Bayes

- XGBoost

 

### Sampling Techniques Compared

- No Sampling

- Random Undersampling

- Random Oversampling

- SMOTE (Synthetic Minority Oversampling Technique)

 

### Scaling Techniques Compared

- No Scaling

- Standard Scaler

- Min-Max Scaler

- Robust Scaler

 

### Hyper Parameters Compared

 

#### Logistic Regression:

- Regularization strength (C): [0.1, 1.0, 10] - The smaller the value the stronger the regularization (Default: 1.0)

- Penalty: [L2] - Type of regularization applied to the model (Default: L2). L1 is excluded

due to lbfgs incompatibility

- Solver: [liblinear, lbfgs] - Algorithm used to optimize the parameters (Default: lbfgs)

- Class weight: [None, balanced] - Balanced automatically adjust the weights inversely proportional to class frequencies (Default: None)

#### Support Vector Classifier:

- Regularization strength (C): [0.1, 1.0, 10] - The smaller the value the stronger the regularization (Default: 1.0)

- Kernel: [Linear, RBF, Sigmoid] - Kernel type for used in algorithm (Default: RBF)

- Gamma: [scale, auto, 0.1, 1.0] - Kernel coefficient (Default: Scale)

- Class weight: [None, balanced] - Balanced automatically adjust the weights inversely proportional to class frequencies (Default: None)

#### Decision Tree Classifier:

- Criterion: [gini, entropy, log_loss] - The function used to measure the quality of a split

(Default: gini)

- Max depth: [None, 5, 10, 15] - Maximum depth of the tree (Default: None)

- Min samples split: [2, 10, 20] - The minimum number of samples required to split an internal

node (Default: 2)

6

- Class weight: [None, balanced] - Balanced automatically adjust the weights inversely proportional to class frequencies (Default: None)

#### Random Forest Classifier:

- N estimators: [100, 200, 300] - Number of trees in the forest (Default: 100)

- Criterion: [gini, entropy, log_loss] - The function used to measure the quality of a split

(Default: gini)

- Max depth: [None, 5, 10, 15] - Maximum depth of the tree (Default: None)

- Min samples split: [2, 10, 20] - The minimum number of samples required to split an internal

node (Default: 2)

- Max features: [None,sqrt, log2] - The number of features to consider when looking for the

best split (Default SQRT)

- Class weight: [None, balanced] – class balancing mechanism

#### Gaussian Naive Bayes Classifier:

- Var smoothing: [1 × 10−9

, 1 × 10−7

, 1 × 10−5

] - Artificially adding a value to the variance

of each feature, widening the distribution and accounting for more samples further from the

mean (Default: 1 × 10−9

)

#### XGBoost Classifier:

- N estimators: [100, 200, 300] - Number of trees in the forest (Default: 100)

- Learning rate: [0.1, 0.2, 0.3] - Step size shrinkage (Default: 0.3)

- Max depth: [3, 6] - Maximum depth of the trees (Default: 6)

- Subsample: [0.8, 1.0] - The fraction of samples to be randomly sampled to build each tree

(Default: 1.0)

- Colsample bytree: [0.8, 1.0] - The fraction of features to be randomly sampled for building

each tree (Default: 1.0)

 

### Performance Metrics Compared

- Precision

- Recall

- F1 Score (Primary metric used to pick optimal configurations for each classification model type)

- ROC AUC

 

---

 

## Datasets

 

1. **Dataset 1**: IBM Telco Churn dataset (7032 rows, 27% churn)

2. **Dataset 2**: A copy of Dataset 1 + 12 engineered features (7032 rows, 27% churn)

3. **Dataset 3**: Public telecom churn dataset (64,374 rows, 47% churn)

4. **Dataset 4**: Banking churn dataset (10,000 rows, 20% churn)

 

---

 

## Results

 

### Optimal Model Configuration (For the Original Telecommunications Dataset)

- **Model**: Support Vector Classifier

- **Scaling**: Standard Scaler

- **Sampling**: Random Oversampling

- **Hyper-Parameters**:

  - **class_weight**:

  - **gamma**:

  - **kernel**:

 

- **Performance Metrics**

  - **'Accuracy'**: 0.93

  - **'Precision'**: 0.96

  - **'Recall'**: 0.90

  - **'F1 Score'**: 0.93 (Primary metric used to pick optimal configurations for each classification model type)

  - **'ROC AUC Score'**: 0.96

 

### Generalization Performance

- **Dataset 2**: Performance dropped across all models for the dataset with the additional engineered features.

- **Dataset 3**: Performance improved significantly (F1 > 0.94 for most models) for the similar dataset also in the telecommuncations industry.

- **Dataset 4**: Performance dropped across all models for the banking dataset, indicating limited cross-industry generalizability.

 <img width="1698" height="670" alt="f1_performance" src="https://github.com/user-attachments/assets/25b51368-4adf-49c1-b9e9-1614015d0621" />


---

 

## SHAP Analysis

 

SHAP was used to interpret the XGBoost model's predictions on each dataset. Below are the top features (derived from mean absolute SHAP values) influencing churn from the original dataset being analyzed:

- Number of Referrals

- Month-to-Month Contract (indicator feature)

- Monthly Charges

- Number of Dependants

- Tenure

- Dependents (indicator feature)

- Age

- Total Charges

- Avg Monthly Long Distance Charges

- Total Revenue

 <img width="990" height="590" alt="image" src="https://github.com/user-attachments/assets/3fc65585-3c6b-4458-aa19-9be3b375eaf1" />


---

 

## Key Takeaways

 

- XGBoost, Random Forest, and Naive Bayes classifiers all had

little variance between combinations of sampling and scaling techniques. Support Vector Classifier, Logistic Regression, and the Decision Tree Classifier all had a significantly higher variance in the ROC Curves given different combinations.

- Hyperparameter tuning significantly boosts performance beyond optimizing for scaling and sampling technique combinations.

- The optimized models that we identified provided mixed results when applied to different datasets. These model configurations may not be as applicable to other industries or datasets, but the process itself may be used to identify the best configurations for a specific dataset.
