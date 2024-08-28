<div align="center">
  <h1>🚀Microsoft: Classifying Cybersecurity Incidents with Machine Learning</h1>
</div>


<p align="center">
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen" alt="Python Badge">
  </a>
  <a href="https://scikit-learn.org/stable/">
    <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn Badge">
  </a>
  <a href="https://numpy.org">
    <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy Badge">
  </a>
  <a href="https://pandas.pydata.org">
    <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas Badge">
  </a>
  <a href="https://plotly.com">
    <img src="https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly Badge">
  </a>
  <a href="https://www.google.com/">
    <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Machine Learning Badge">
  </a>
</p>


<p align="center">
  <img width="750" height="300" alt="dictionary" src="https://github.com/user-attachments/assets/1d9a0498-0f2e-475d-be2d-4586719d6af6">
</p>

## 📂 Project Overview

In this project, we aim to assist Security Operations Centers (SOCs) by building a machine learning model that predicts triage grades for cybersecurity incidents. These incidents will be categorized as **True Positive (TP)**, **Benign Positive (BP)**, or **False Positive (FP)** using the **GUIDE** dataset from Microsoft. The model's predictions will help SOC analysts make quicker, data-driven decisions, improving threat detection accuracy and response times in enterprise environments.

---

## 🎯 Project Scope

This project encompasses the entire machine learning pipeline, from Data Preprocessing to Evaluation. The main components of the scope include:

1. **Data Exploration & Preprocessing**: Conducting exploratory data analysis (EDA) to understand patterns and trends in the dataset. This includes handling missing data, feature engineering, and transforming categorical variables.

2. **Model Development**: Building a classification model using various machine learning techniques. Initial models will be developed as baselines, and advanced models like Random Forests and XGBoost will be used for optimization.

3. **Addressing Imbalanced Data**: Implementing methods like **SMOTE** or **class weight adjustments** to handle imbalanced datasets effectively.

4. **Evaluation & Interpretation**: Evaluating the model's performance using metrics such as **Macro-F1 score**, **Confusion Matrix**, **Precision** & **Recall**. Additionally, model interpretation techniques like SHAP values will be used to determine the importance of features.

---

## 🧠 Problem Statement

The task is to design a machine learning model that can enhance the SOC's triage process by accurately classifying cybersecurity incidents. The model should be able to predict whether an incident is a True Positive (TP), Benign Positive (BP), or False Positive (FP) based on historical evidence and customer responses. 

The challenge lies in the complexity of cybersecurity data and the need to minimize false positives while ensuring that real threats are correctly identified. By automating this classification process, the project aims to reduce the burden on SOC analysts and improve response times for critical security threats, thereby improving the overall security posture of enterprise environments.

---

## 💡 Dataset Overview

We will be utilizing two datasets, __train__ and __test__, for our analysis. Both datasets contain over 9.5 million rows of data, and data preprocessing will be applied to both to ensure consistency and accuracy in the model's performance.

The **GUIDE** dataset is structured into three hierarchies:

1. **Evidence Level**: IP addresses, emails, user details.
2. **Alert Level**: Consolidation of evidence to signify potential security incidents.
3. **Incident Level**: Comprehensive security breach narratives representing a cohesive threat scenario.

The dataset includes **45 features** and over **1 million incidents** with triage annotations (**TP, BP, FP**). It is divided into **70% training data** and **30% test data**.

---

## 📊 Metrics

Predicting the triage grade of cybersecurity incidents is a __classification problem__. Therefore, the following metrics that are essential for classification problems are taken into account. Below are the __metrics__ used in the process of classifying cybersecurity incidents:

- [__Macro-F1 Score__](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f): This metric provides a balanced measure of performance across all classes by averaging the F1 scores of each class, treating all classes equally.

- [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#precision-score): This metric measures the accuracy of the positive predictions made by the model, focusing on how many of the predicted positives are actually true positives.

- [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#recall-score): This metric assesses the model's ability to identify all relevant instances (true positives), focusing on how many of the actual positives were correctly predicted.


---

 ## 🛡️Business Use Cases

The solution developed in this project can be applied to various business scenarios in cybersecurity:

- 🏢 **Security Operation Centers (SOCs)**: Automate the triage process by accurately classifying incidents, allowing SOC analysts to prioritize and respond to critical threats more efficiently.

- 🤖 **Incident Response Automation**: Enable guided response systems to automatically suggest actions for different types of incidents, leading to quicker threat mitigation.

- 🔍 **Threat Intelligence**: Enhance threat detection by incorporating historical evidence and customer responses, leading to more accurate identification of true and false positives.

- 🚀 **Enterprise Security Management**: Improve security posture by reducing false positives and ensuring true threats are addressed promptly.

---

# 🛠️ Approach

## 1. Data Exploration and Understanding 🔍

### a. Initial Inspection 🕵️‍♂️
- **Objective**: Understand the dataset's basic structure and features.
- **Steps**:
  - Load the `train.csv` dataset using appropriate tools (e.g., pandas in Python).
  - Inspect the dataset to check:
    - The number of rows and columns.
    - Types of features (categorical, numerical).
    - Summary statistics of numerical features (mean, median, standard deviation).
    - The distribution of the target variable (TP, BP, FP) to assess class balance.

### b. Exploratory Data Analysis (EDA) 📉
- **Objective**: Discover patterns, relationships, and anomalies in the data.
- **Steps**:
  - Generate visualizations such as histograms, scatter plots, and box plots to understand data distributions and relationships.
  - Compute statistical summaries to quantify feature distributions.
  - Identify and document any class imbalances, which may necessitate specialized handling methods.

## 2. Data Preprocessing 🛠️

### a. Handling Missing Data 🚫
- **Objective**: Address missing values to prepare the dataset for modeling.
- **Steps**:
  - Identify missing values using functions like `.isnull()` in pandas.
  - Decide on a strategy for each column with missing data:
    - **Imputation**: Replace missing values with mean, median, or mode.
    - **Removal**: Drop rows or columns with missing values if they are not critical.
    - **Model-based**: Use algorithms that can handle missing values directly, if applicable.

### b. Feature Engineering ✨
- **Objective**: Enhance the dataset by creating or modifying features to improve model performance.
- **Steps**:
  - **Combine Features**: Merge related features (e.g., combining `year` and `month` into a `date` feature).
  - **Derive New Features**: Extract new information from existing features (e.g., converting timestamps into hour of the day or day of the week).
  - **Normalization**: Scale numerical features to a common range (e.g., using Min-Max scaling or Z-score normalization).

### c. Encoding Categorical Variables 🔢
- **Objective**: Convert categorical features into a numerical format suitable for modeling.
- **Steps**:
  - **One-Hot Encoding**: Create binary columns for each category (e.g., using `pd.get_dummies()`).
  - **Label Encoding**: Assign integer values to each category (e.g., using `LabelEncoder`).
  - **Target Encoding**: Replace categories with the mean of the target variable for that category.

## 3. Data Splitting 🧩

### a. Train-Validation Split 🧪
- **Objective**: Divide the dataset into training and validation sets to evaluate model performance.
- **Steps**:
  - Split the `train.csv` data into:
    - **Training Set**: Used for training the model.
    - **Validation Set**: Used for tuning and evaluating model performance.
  - Common splits include 70-30 or 80-20, but this can vary based on the dataset size.

### b. Stratification ⚖️
- **Objective**: Ensure that both training and validation sets have similar class distributions.
- **Steps**:
  - If the target variable is imbalanced, use stratified sampling to maintain proportional class distribution in both sets.
  - This helps in avoiding bias in the model evaluation process.

## 4. Model Selection and Training 🏗️

### a. Baseline Model 🚀
- **Objective**: Establish a performance benchmark with a simple model.
- **Steps**:
  - Start with basic models like:
    - **Logistic Regression**: For binary classification problems.
    - **Decision Tree**: For a simple tree-based approach.
  - Evaluate the baseline model's performance to understand the initial effectiveness and complexity needed.

### b. Advanced Models 🔍
- **Objective**: Explore more sophisticated models to improve performance.
- **Steps**:
  - Experiment with advanced models such as:
    - **Random Forests**: Ensemble method that uses multiple decision trees.
    - **Gradient Boosting Machines (e.g., XGBoost, LightGBM)**: Boosting algorithms that build models sequentially.
    - **Neural Networks**: For capturing complex patterns with deep learning.
  - Use techniques like grid search or random search for hyperparameter tuning.

### c. Cross-Validation 🔄
- **Objective**: Validate model performance across different data subsets.
- **Steps**:
  - Implement k-fold cross-validation:
    - Divide the data into k subsets (folds).
    - Train and evaluate the model k times, each time using a different fold as the validation set and the remaining as training data.
  - This reduces the risk of overfitting and provides a more reliable estimate of model performance.

## 5. Model Evaluation and Tuning 🧪

### a. Performance Metrics 📈
- **Objective**: Assess model performance using relevant metrics.
- **Steps**:
  - Evaluate using the validation set:
    - **Macro-F1 Score**: Measures balance between precision and recall across classes.
    - **Precision**: Accuracy of positive predictions.
    - **Recall**: Ability to identify all positive instances.
  - Analyze metrics to ensure balanced performance across all classes (TP, BP, FP).

### b. Hyperparameter Tuning ⚙️
- **Objective**: Optimize model parameters to enhance performance.
- **Steps**:
  - Adjust hyperparameters like learning rates, regularization terms, tree depths, or number of estimators.
  - Use grid search or random search methods to find the best parameter combination.

### c. Handling Class Imbalance ⚠️
- **Objective**: Address any issues with imbalanced class distributions.
- **Steps**:
  - Implement techniques such as:
    - **SMOTE (Synthetic Minority Over-sampling Technique)**: Generate synthetic samples for minority classes.
    - **Class Weights**: Adjust weights in the model to compensate for class imbalance.
    - **Ensemble Methods**: Use techniques like bagging or boosting to improve performance on minority classes.

## 6. Model Interpretation 🔎

### a. Feature Importance 🌟
- **Objective**: Understand the contribution of each feature to the model's predictions.
- **Steps**:
  - Analyze feature importance using:
    - **SHAP Values**: Measure the impact of each feature on the model’s output.
    - **Permutation Importance**: Assess how performance changes when feature values are permuted.
    - **Model-Specific Methods**: Use feature importance scores provided by models like Random Forest.

### b. Error Analysis 📝
- **Objective**: Identify and understand common misclassifications.
- **Steps**:
  - Analyze errors to identify patterns or trends.
  - Use insights to refine feature engineering or adjust model complexity.

## 7. Final Evaluation on Test Set 🧾

### a. Testing 🧪
- **Objective**: Evaluate the finalized model on unseen data.
- **Steps**:
  - Test the model using the `test.csv` dataset.
  - Report final performance metrics: macro-F1 score, precision, and recall.

### b. Comparison to Baseline 📊
- **Objective**: Verify model improvement and consistency.
- **Steps**:
  - Compare performance on the test set with the baseline model and validation results.
  - Ensure that the final model shows consistent improvement and meets project objectives.

## 8. Model Comparison 🔍

|  __Model__  |  __Macro-F1 Score__  | __Precision__ | __Recall__ | 
| :-: | :-: | :-: | :-: |
| [__1. Logistic Regression__](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | 0.845 | 0.820 | 0.870 | 
| [__2. Random Forest Classifier__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | 0.860 | 0.830 | 0.890 |
| [__3. Gradient Boosting Machines__](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) | 0.855 | 0.825 | 0.880 | 

* The Random Forest Classifier shows the best overall performance with the highest Macro-F1 Score, indicating superior classification capability.

---

## 🏆 Results

By the conclusion of this project, the following outcomes are expected:

- **Accurate Machine Learning Model**: Develop a machine learning model that reliably predicts the triage grade of cybersecurity incidents (True Positive, Benign Positive, False Positive) with a high macro-F1 score, precision, and recall.
  
- **Comprehensive Performance Analysis**: Provide a detailed analysis of the model's performance, including insights into the most influential features in the prediction process.

---





