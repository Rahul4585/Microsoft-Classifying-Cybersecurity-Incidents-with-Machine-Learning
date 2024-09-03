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

Our project encompasses the entire machine learning pipeline, from Data Preprocessing to Evaluation. The main components of the scope include:

1. **Data Exploration & Preprocessing**: Conducting exploratory data analysis (EDA) to understand patterns and trends in the dataset. This includes handling missing data, feature engineering, and transforming categorical variables.

2. **Model Development**: Building a classification model using various machine learning techniques. Initial models will be developed as baselines, and advanced models like Random Forests and XGBoost will be used for optimization.

3. **Evaluation & Interpretation**: Evaluating the model's performance using metrics such as **Accuracy**, **Macro-F1 score**, **Confusion Matrix**, **Precision** & **Recall**. Additionally, model interpretation techniques will be used to determine the importance of features.

---

## 🧠 Problem Statement

The challenge was to automate the triage process by developing a machine learning model to accurately classify cybersecurity incidents as TP, BP, or FP based on historical data. Reducing false positives while ensuring critical threats are not missed was key to improving SOC efficiency.

---

## 💡 Dataset Overview

We will be utilizing two datasets, __train__ and __test__, for our analysis. Both datasets contain over 9.5 million rows of data, and data preprocessing was applied to both datasets to ensure consistency and accuracy in the model's performance.

You can download the dataset from the following link:

[Download Dataset](https://drive.google.com/drive/folders/18vt2lkf69MggXitrTSn9qnZ8s-ToeKcH)

The dataset is structured into three hierarchies:

1. **Evidence Level**: Includes IPs, emails, and user information.
2. **Alert Level**: Summarizes evidence into potential security incidents..
3. **Incident Level**: Provides narratives of comprehensive security threats.

The dataset was split into **70% training data** and **30% test data**.

---

## 📊 Metrics

We are predicting the triage grade of cybersecurity incidents. Therefore, the following __metrics__ that we used for classification are :

- [__Accuracy__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html): Measures the overall correctness of the predictions.

- [__Macro-F1 Score__](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f): Balances performance across all classes by averaging F1 scores.

- [__Precision__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#precision-score): Measures how many of the positive predictions were correct.

- [__Recall__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#recall-score): Evaluates the ability of the model to capture all relevant positive cases.

- [__Confusion Matrix__](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html):Visual representation of model performance across classes.

---

 ## 🛡️Business Use Cases

The solution developed in our project can be applied to various business scenarios in cybersecurity:

- **Security Operation Centers (SOCs)**: Automate the triage process by accurately classifying incidents, allowing SOC analysts to prioritize and respond to critical threats more efficiently.

- **Incident Response Automation**: Enable guided response systems to automatically suggest actions for different types of incidents, leading to quicker threat mitigation.

- **Threat Intelligence**: Enhance threat detection by incorporating historical evidence and customer responses, leading to more accurate identification of true and false positives.

- **Enterprise Security Management**: Improve security posture by reducing false positives and ensuring true threats are addressed promptly.

---

# 🛠️ Approach

## 1. Data Exploration and Understanding

### a. Initial Inspection
- Loaded `train.csv` and inspected the dataset:
  - Verified row/column counts.
  - Analyzed feature types (categorical, numerical).
  - Checked summary statistics of numerical features.
  - Evaluated the target class distribution (TP, BP, FP).

### b. Exploratory Data Analysis (EDA)
- **Objective**: Discover patterns, relationships, and anomalies in the data.

  - We have generated visualizations,  to understand data distributions and relationships.

 **i) Class Distribution**

* This graph helps to understand how balanced or imbalanced the `IncidentGrade` classes are within the dataset.

<p align="center">
  <img width="700" alt="Class Distribution" src="https://github.com/user-attachments/assets/df346449-468a-4719-9142-152a2969aa8d">
</p>

**ii) Correlation Heatmap of Numerical Features**

* The heatmap is used to examine the relationships between numerical features in the dataset.

<p align="center">
  <img width="700" alt="Correlation Heatmap" src="https://github.com/user-attachments/assets/b65453a3-dddc-4e65-b534-55942c98ce44">
</p>

**iii) Distribution of Suspicion Levels**

* This graph is used to analyze how different levels of suspicion are distributed in the dataset.

<p align="center">
  <img width="700" alt="Distribution of Suspicion Levels" src="https://github.com/user-attachments/assets/89dd1cd0-0550-4754-b2b5-cd1488788d24">
</p>

**iv) Distribution of Incidents by Hour**

* This histogram illustrates the frequency of incidents occurring at different hours of the day.

<p align="center">
  <img width="700" alt="Distribution of Incidents by Hour" src="https://github.com/user-attachments/assets/acb33e3a-602a-4a64-a466-06163ab2dcdc">
</p>

**v) Correlation Heatmap for a Subset of Columns**

* The subset correlation heatmap is used to analyze the relationships between specific columns in the dataset.

<p align="center">
  <img width="700" alt="Subset Correlation Heatmap" src="https://github.com/user-attachments/assets/64a7b779-d3dc-4706-9877-1ec11e5b722f">
</p>

## 2. Data Preprocessing

### a. Handling Missing Data
- **Objective**: Missing values are addressed to prepare the dataset for modeling.

  - Identified missing values using functions like `.isnull()` in pandas.
  - For each columns with missing data we have used the following methods:
    - **Imputation**: Replaced missing values with mean, median, and mode.
    - **Removal**: Dropped rows and columns with missing values .
    - **Model-based**: Used algorithms that can handle missing values directly.

### b. Feature Engineering
- **Objective**: We have enhance the dataset by modifying the features to improve model performance.

  - **Derive New Features**: Extracted new information from existing features such as converting timestamps into hour of the day or day of the week.

### c. Encoding Categorical Variables
- **Objective**: Converted categorical features into a numerical format suitable for modeling.

  - **One-Hot Encoding**: We have created binary columns for each category by using `pd.get_dummies()`.
  - **Label Encoding**: Assigned integer values to each category using `LabelEncoder`).
   
## 3. Data Splitting

### a. Train-Validation Split
- **Objective**: Divided the dataset into training and validation sets to evaluate model performance.

  - Splited the `train.csv` data into:
    - **Training Set**: Used for training the model.
    - **Validation Set**: Used for tuning and evaluating model performance.
  - We have split the data into 70-30 ratio for train and test respectively.

## 4. Model Selection and Training

### a. Baseline Models
- **Objective**: Established a performance benchmark using a simple model to gauge the complexity required for the final model.
 
   - **i) Logistic Regression**: Chosen for its simplicity and effectiveness in binary classification tasks. It provides a good starting point to understand the relationship between the features and the target variable.

<p align="center">
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/2788ab95-703c-4ec4-84aa-58a46cfd1474">
<p align="center">

   - **ii) Decision Tree**: Selected as an alternative baseline model, offering interpretability and the ability to capture non-linear relationships in the data.

<p align="center">
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/9f581f26-995b-4e57-86bc-c8268ab64fc3">
<p align="center">

### b. Advanced Models
- **Objective**: Explored more sophisticated models to improve performance.

   - **i) Random Forests**: An ensemble learning method that constructs a multitude of decision trees during training. The final prediction is made by averaging the predictions of individual trees (for regression) or by majority voting (for classification).
           
<p align="center">
  <img width="700" alt="Image" src="https://github.com/user-attachments/assets/946bd8b2-87e1-46a1-ad79-1046cc151c8b">
</p>

   - **ii) Gradient Boosting Machines (e.g., XGBoost, LightGBM)**:
     
     - **Description**: Gradient Boosting Machines (GBMs) are boosting algorithms that build models sequentially. Each new model attempts to correct errors made by the previous ones, resulting in a strong predictive model. XGBoost (Extreme Gradient Boosting) and LightGBM (Light Gradient Boosting Machine) are popular implementations that improve upon traditional boosting methods with enhanced speed and accuracy.
       
<p align="center">
  <img width="700" alt="Image" src="https://github.com/user-attachments/assets/b629c2be-b9dd-4d77-b156-ea84652c2366">
</p>

### c. Cross-Validation
- **Objective**: To validate model performance across different data subsets.

  - We have implemented k-fold cross-validation:
    - Divided the data into k subsets (folds).
    - Trained and evaluated the model k times, each time used a different fold as the validation set and the remaining as training data.
  - This has reduced the risk of overfitting and provided a more reliable estimate of model performance.

## 5. Model Evaluation and Tuning

### a. Performance Metrics
- **Objective**: To assess model performance using relevant metrics.

  - Evaluate using the validation set:
    - **Accuracy**: Measured the overall correctness of the predictions.  
    - **Macro-F1 Score**: Measured the balance between precision and recall across classes.
    - **Precision**: Accuracy of positive predictions.
    - **Recall**: Ability to identify all positive instances.
    - **Confusion Matrix**: Visual representation of model performance across classes.
  - Analyze metrics to ensure balanced performance across all classes (TP, BP, FP).

### b. Hyperparameter Tuning
- **Objective**: Optimized the model parameters to enhance performance.

  - Adjusted hyperparameters like learning rates, regularization terms, tree depths, or number of estimators.
  - Used grid search method to find the best parameter combination.

## 6. Model Interpretation

### a. Feature Importance
- **Objective**: Understanding the contribution of each feature to the model's predictions.

  - Analyzed the feature importance using:
    - **Model-Specific Methods**: We have used feature importance scores provided by models like Random Forest.
   
i) __Feature Importances (Random Forest Classifier)__

<p align="center">
  <img width="700" alt="Image" src="https://github.com/user-attachments/assets/87daa338-dfd0-4e1b-bfd2-80a18e84f74a">
</p>

ii) __Feature Importances (XGBoost Classifier)__

<p align="center">
  <img width="700" alt="Image" src="https://github.com/user-attachments/assets/a03d3ce6-6c16-475a-8e31-9736d08890fb">
</p>

## 7. Final Evaluation on Test Set

### a. Testing
- **Objective**: Evaluated the finalized model on unseen data.

  - Tested the model using the `test.csv` dataset.
  - Reported final performance metrics: accuracy,macro-F1 score, precision,recall , confusion matrix.

---

## 🔍 Model Comparison

|  **Model**  | **Accuracy** |  **Macro-F1 Score**  | **Macro-Precision** | **Macro-Recall** |
|:-----------:|:------------:|:-------------------:|:-------------------:|:----------------:|
| [**1. Random Forest Classifier**](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) | 0.498 | 0.347 | 0.679 | 0.411 |
| [**2. XGBoost Classifier**](https://xgboost.readthedocs.io/en/stable/) | 0.621 | 0.571 | 0.694 | 0.563 |
| [**3. Logistic Regression**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) | 0.433 | 0.239 | 0.272 | 0.352 |
| [**4. Decision Tree Classifier**](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) | __0.807__ | __0.790__ | __0.796__ | __0.786__|



**Decision Tree Classifier** performs better across all metrics compared to the other Classifiers. It has higher accuracy, Macro-F1 Score, Macro-Precision, and Macro-Recall, making it the superior model based on the provided data.

---

## 📚 Recommendations

### a) Integration into SOC Workflows

- **SIEM Systems:** Integrate the model to enhance alert prioritization and response.
- **Incident Management:** Ensure compatibility for streamlined incident handling.
- **Automated Response:** Use model predictions to automate responses and actions.
- **Real-Time Monitoring:** Implement continuous monitoring and updates.

### b) Future Improvements

- **Feature Expansion:** Explore additional features and advanced techniques.
- **Data Balance:** Address class imbalance with oversampling or undersampling.
- **Explainability:** Enhance model interpretability for better analyst understanding.
- **Feedback Loop:** Integrate analyst feedback for ongoing improvements.

### c) Deployment Considerations

- **Scalability:** Test for performance and scalability in production.
- **Security:** Ensure data protection and privacy compliance.
- **Operationalization:** Establish a deployment pipeline with testing and validation.
- **Training:** Provide SOC analyst training and support.
- **Maintenance:** Set up ongoing evaluation and maintenance for model performance.
 
---

## 🏆 Results

By the conclusion of this project, the following outcomes are incorprated:

- **Accurate Machine Learning Model**: Developed a machine learning model that reliably predicted the triage grade of cybersecurity incidents (True Positive, Benign Positive, False Positive) with high performance metrics, including macro-F1 score, precision, and recall. The **Decision Tree Classifier** is anticipated to achieve the best results.

- **Comprehensive Performance Analysis**: Provided a detailed analysis of the model's performance, including insights into the most influential features in the prediction process. This will involve evaluating feature importance, understanding the impact of various features on the model’s predictions, and identifying areas for potential improvement.

---
