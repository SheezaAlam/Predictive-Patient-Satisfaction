[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green)]()
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Hospital Patient Satisfaction Prediction

Machine Learning Project • University Assignment

This project predicts patient satisfaction levels in a hospital environment using machine-learning classification algorithms. The workflow follows the required assignment structure: data loading, preprocessing, feature engineering, model training, model comparison, visualization, and conclusion.

---

## Project Overview

Hospitals collect data regarding patient demographics, services provided, staffing levels, and operational events.
The goal of this project is to use these features to predict whether a patient reports high satisfaction (defined as satisfaction ≥ 80).

The following five classification models are implemented:

* Logistic Regression
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)

To handle class imbalance, SMOTE (Synthetic Minority Oversampling Technique) is applied.

---

## Dataset Description

The project uses four datasets:

| File Name             | Description                                         |
| --------------------- | --------------------------------------------------- |
| `patients.csv`        | Patient demographic and satisfaction data           |
| `services_weekly.csv` | Weekly service-level metrics (beds, morale, events) |
| `staff_schedule.csv`  | Staff presence by week, service, and role           |
| `staff.csv`           | Staff details                                       |

These files are automatically loaded from the folder:

```
~/Downloads/archive (1)/
```

The path may be adjusted if the dataset is stored elsewhere.

---

## Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Machine Learning Pipelines

---

## Steps in the Project

### 1. Data Loading

All CSV files are loaded, checked for shape, and validated for correctness.

### 2. Data Exploration

Initial analysis includes:

* Displaying sample rows
* Checking missing values
* Inspecting data types
* Creating the binary target variable:

```
high_satisfaction = 1 if satisfaction >= 80 else 0
```

### 3. Feature Engineering

New features include:

* Length of stay
* Week number
* Staff counts aggregated by week, service, and role
* Service-level metrics merged into the patient dataset

Features are categorized as:

* Numerical: age, length_of_stay, available_beds, patients_refused, staff_morale, doctor, nurse, nursing_assistant
* Categorical: service, event

### 4. Preprocessing

A pipeline is created that applies:

* Standard scaling to numerical features
* One-hot encoding to categorical features
* SMOTE to address class imbalance
* Model fitting

### 5. Model Training and Evaluation

Each model undergoes:

* 5-fold cross-validation (F1-score as metric)
* Training on the training set
* Evaluation on the test set
* Confusion matrix visualization
* ROC curve plotting (where applicable)

Metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

---

## Model Comparison

A final results table compares all models.

Random Forest achieves the highest F1-score and ROC-AUC, making it the best-performing model for this dataset.

---

## Conclusion

Random Forest achieved the strongest performance among all evaluated models, with high F1 and ROC-AUC scores. Its ensemble structure helps capture non-linear relationships between variables such as staff presence, bed availability, patient length of stay, and departmental events. This makes Random Forest the most reliable model for predicting patient satisfaction in real-world hospital settings.

Logistic Regression and SVM performed reasonably well but were limited due to their linear assumptions. The Decision Tree model showed overfitting tendencies, and KNN, while stable, did not outperform the Random Forest.
Overall, Random Forest is recommended for deployment in hospital quality control and satisfaction prediction systems.

---

## How to Run the Project

1. Install the required dependencies:

```
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

2. Place the dataset folder at:

```
~/Downloads/archive (1)/
```

3. Run the Jupyter Notebook:

```
jupyter notebook
```

4. Execute all cells in sequence.

---

## Acknowledgements

This project was developed as part of a university assignment focusing on:

* Data preprocessing
* Handling multiple datasets
* Working with class imbalance
* Evaluating and comparing machine learning classification models

---

