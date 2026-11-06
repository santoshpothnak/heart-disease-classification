# Heart Disease Classification and Streamlit Cloud Deployment

## a. Problem Statement
Heart disease remains one of the leading global causes of mortality. Early, accurate detection allows clinicians to prioritize timely interventions and reduce adverse cardiovascular outcomes. The objective of this project is to develop, evaluate, and benchmark five distinct supervised machine learning classification algorithms to predict the presence of cardiovascular disease from clinical physiological indicators. The benchmarked models are deployed to an interactive Streamlit web dashboard where external test records can be uploaded and evaluated without retraining.

## b. Dataset Description
- **Source**: UCI Machine Learning Repository / Kaggle Heart Disease Clinical Dataset
- **Total Instances**: 1,025 patient records (conforms to the minimum requirement of >= 500)
- **Total Input Features**: 13 diagnostic parameters (conforms to the minimum requirement of >= 12)
- **Target Variable**: `target` (Binary classification: 0 = Absence of Heart Disease, 1 = Presence of Heart Disease)

### Feature Attributes
1. `age`: Patient age in years
2. `sex`: Biological sex (1 = male, 0 = female)
3. `cp`: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)
4. `trestbps`: Resting blood pressure in mm Hg on admission to the hospital
5. `chol`: Serum cholesterol measurement in mg/dl
6. `fbs`: Fasting blood sugar level > 120 mg/dl (1 = true, 0 = false)
7. `restecg`: Resting electrocardiographic measurement (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)
8. `thalach`: Maximum heart rate achieved during exercise stress testing
9. `exang`: Exercise-induced angina (1 = yes, 0 = no)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
12. `ca`: Number of major blood vessels (0-3) colored by fluoroscopy
13. `thal`: Thalassemia stress test result (1 = normal, 2 = fixed defect, 3 = reversible defect)

## c. GitHub Repository Link
- **GitHub Repository**: `https://github.com/santoshpothnak/heart-disease-classification`
- **Streamlit Live App**: `https://santoshpothnak-heart-disease-classification.streamlit.app`

## d. Models Used & Evaluation Metrics
All models were evaluated on the held-out 20% test partition (205 records) using stratified sampling (`random_state=42`).

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8244 | 0.8938 | 0.8115 | 0.8839 | 0.8462 | 0.6457 |
| **Decision Tree** | 0.8829 | 0.9325 | 0.8548 | 0.9464 | 0.8983 | 0.7667 |
| **KNN** | 0.9902 | 0.9994 | 0.9825 | 1.0000 | 0.9912 | 0.9805 |
| **Naive Bayes** | 0.8098 | 0.8927 | 0.8174 | 0.8393 | 0.8282 | 0.6154 |
| **Support Vector Machine** | 0.9220 | 0.9610 | 0.9138 | 0.9464 | 0.9298 | 0.8426 |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Provided a solid linear baseline (Accuracy: 0.8244, AUC: 0.8938). It achieved strong recall (0.8839), ensuring few false negatives, but exhibited slightly lower precision (0.8115) due to its inability to capture non-linear interactions among clinical attributes like `oldpeak` and `thalach`. |
| **Decision Tree** | Strong recall (0.9464) and solid precision (0.8548) with an accuracy of 0.8829 and AUC of 0.9325. Setting `max_depth=5` and `min_samples_split=4` controlled variance and overfitting while capturing multi-tier condition thresholds (e.g., chest pain type and fluoroscopy vessels). |
| **KNN** | Delivered top-tier performance across all metrics (Accuracy: 0.9902, AUC: 0.9994, MCC: 0.9805) using distance-weighted voting ($k=7$) on scaled features. Local neighborhood clustering effectively separated clinical profiles, achieving 100% recall on the test partition. |
| **Naive Bayes** | Displayed the lowest comparative performance (Accuracy: 0.8098, MCC: 0.6154). The Gaussian conditional independence assumption was partially violated due to collinearity between exercise-induced angina (`exang`), ST depression (`oldpeak`), and peak heart rate (`thalach`). |
| **Support Vector Machine** | Displayed excellent discriminative capacity (Accuracy: 0.9220, AUC: 0.9610, F1: 0.9298). The non-linear RBF kernel established smooth decision boundaries across complex multidimensional physiological margins, maintaining high recall (0.9464). |
| **Overall, Winner for your dataset?** | **K-Nearest Neighbors (KNN)** (closely supported by **Support Vector Machine (SVM)**). KNN achieved the highest overall Accuracy (0.9902), F1 Score (0.9912), and Matthews Correlation Coefficient (0.9805). For clinical diagnostics, sensitivity/recall is paramount to avoid missing high-risk cardiac patients; KNN achieved a perfect **1.0000 recall** on the held-out test split, making it the most reliable model for patient triage. For production deployment where inference latency under high sample counts matters, SVM serves as an equally robust parametric alternative (AUC: 0.9610). |
