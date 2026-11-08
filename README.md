# Online Shoppers Purchasing Intention Prediction

## 1. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to predict whether an online shopping session will result in a purchase.

Five classification algorithms are implemented and compared using multiple evaluation metrics. A Streamlit application is also developed to allow users to upload test data, select a trained model, and view its evaluation results.

## 2. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The dataset contains 12,330 online shopping sessions and 18 columns, including 17 input features and one target variable (`Revenue`).

The target variable indicates whether a shopping session resulted in a purchase:
- `True`: Purchase
- `False`: No Purchase

The dataset contains numerical and categorical features describing visitor behaviour, browsing activity, operating systems, browser information, traffic sources, and session characteristics.

The dataset contains 125 duplicate records, which were removed during preprocessing, resulting in 12,205 records.

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Inspected the dataset structure and checked for missing values.
- Identified and removed duplicate records.
- Separated the input features and target variable.
- Converted the target variable into binary values.
- Identified numerical and categorical features.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used stratified sampling to preserve the target class distribution.

The random state was set to 42 to ensure reproducibility.

## 4. Machine Learning Models

The following five classification algorithms were implemented:

1. **Logistic Regression:** A classification algorithm that estimates the probability of a binary outcome.
2. **Decision Tree:** A tree-based algorithm that makes predictions using a sequence of decision rules.
3. **K-Nearest Neighbors (KNN):** A classification algorithm that predicts the class based on the nearest training observations.
4. **Gaussian Naive Bayes:** A probabilistic classification algorithm based on Bayes' theorem and the Gaussian distribution assumption.
5. **Support Vector Machine (SVM):** A classification algorithm that identifies a decision boundary that separates classes.

All models use the same preprocessing pipeline and training/testing split.

## 5. Model Evaluation Metrics

The models were evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified observations.
- **AUC:** Measures the ability of the model to distinguish between the two classes.
- **Precision:** Measures the proportion of predicted purchases that were actual purchases.
- **Recall:** Measures the proportion of actual purchases correctly identified.
- **F1-score:** Represents the harmonic mean of precision and recall.
- **MCC:** Matthews Correlation Coefficient measures the quality of binary classifications, considering all four confusion matrix outcomes.

## 6. Model Comparison Results

The following table presents the evaluation results obtained on the held-out test dataset containing 2,441 records.

| Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8873 | 0.8997 | 0.7560 | 0.4136 | 0.5347 | 0.5049 |
| Decision Tree | 0.8546 | 0.7422 | 0.5325 | 0.5785 | 0.5546 | 0.4684 |
| KNN | 0.8746 | 0.7979 | 0.6667 | 0.3979 | 0.4984 | 0.4507 |
| Gaussian Naive Bayes | 0.2773 | 0.7509 | 0.1756 | 0.9791 | 0.2978 | 0.1375 |
| Support Vector Machine | 0.8951 | 0.8726 | 0.7368 | 0.5131 | 0.6049 | 0.5586 |

## 7. Observations and Conclusion

**Logistic Regression:** Achieved the highest AUC score of 0.8997 and the highest precision of 0.7560. It provides good class discrimination but has comparatively lower recall.

**Decision Tree:** Achieved a recall of 0.5785, performing better than Logistic Regression and KNN in identifying actual purchase sessions. However, its overall accuracy and AUC were lower.

**KNN:** Achieved an accuracy of 0.8746 but had a comparatively lower F1-score of 0.4984.

**Gaussian Naive Bayes:** Achieved the highest recall of 0.9791, but its accuracy was only 0.2773 and precision was 0.1756. This indicates a high number of false-positive predictions.

**Support Vector Machine:** Achieved the highest accuracy (0.8951), F1-score (0.6049), and MCC (0.5586) among the five models.

**Overall Conclusion:**

Based on the comparison of all six evaluation metrics, Support Vector Machine (SVM) was selected as the overall best-performing model for this experiment. It achieved the highest accuracy, F1-score, and MCC, indicating the strongest overall balance among the evaluated models. Logistic Regression performed best in terms of AUC and precision.

However, model selection may depend on the application objective, particularly the trade-off between identifying actual purchases and avoiding false-positive predictions.

## 8. Streamlit Application

A Streamlit-based web application was developed to demonstrate the trained classification models.

### Application Features

- Upload a test dataset in CSV format.
- Preview the uploaded dataset.
- Select one of the five trained classification models.
- Evaluate the selected model.
- Display Accuracy, AUC, Precision, Recall, F1-score, and MCC.
- Display the confusion matrix.
- Display the classification report.
- Compare the performance of all five models.

The application loads the previously trained and saved models. It does not retrain the models when evaluating uploaded test data.


## 9. Project Structure

```text
Online-Shoppers-Purchasing-Intention-ML/
│
├── .devcontainer/
├── assets/
├── models/
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
│
├── app.py
├── Online_Shoppers_ML_Assignment.ipynb
├── test_data.csv
├── requirements.txt
└── README.md
```

## 10. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/Sravyateja/Online-Shoppers-Purchasing-Intention-ML.git
```

Navigate to the project directory:

```bash
cd Online-Shoppers-Purchasing-Intention-ML
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload `test_data.csv` through the application to evaluate the trained models.


## 11. Repository and Application Links

**GitHub Repository:**

https://github.com/Sravyateja/Online-Shoppers-Purchasing-Intention-ML

**Live Streamlit Application:**

https://online-shoppers-purchasing-intention-ml.streamlit.app/


## 12. Execution Environment

The machine learning implementation and model evaluation were performed using Python in the BITS Virtual Lab environment.

The trained models were saved and subsequently integrated into the Streamlit application for evaluation using the held-out test dataset.

The following key library versions were used:

- Python: As provided in BITS Virtual Lab
- NumPy: 1.26.4
- SciPy: 1.13.1
- Scikit-learn: 1.5.2

## 13. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

---

**Academic Programme:** M.Tech in Artificial Intelligence and Machine Learning

**Institution:** BITS Pilani Wilp

**Assignment:** Machine Learning Assignment 2
