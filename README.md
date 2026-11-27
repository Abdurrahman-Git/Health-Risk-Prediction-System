# Multi-Disease Health Risk Prediction System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-00599C?style=for-the-badge)
![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-009688?style=for-the-badge)
![SHAP](https://img.shields.io/badge/Explainable_AI-8A2BE2?style=for-the-badge)

## Project Overview

An end-to-end machine learning application designed to estimate the likelihood of three chronic health conditions:

- Diabetes
- Heart Disease
- High Blood Pressure

The system integrates predictive analytics, exploratory data analysis, explainable artificial intelligence and general health recommendations into a unified, interactive Streamlit web application.

This project was developed as part of the BSc (Hons) Data Science final year project at London Metropolitan University.

## Project Highlights

| Component | Details |
|---|---|
| Dataset | 253,680 health survey records |
| Prediction targets | 3 chronic diseases |
| Trained models | 9 machine learning classifiers |
| Selected models | 3 disease-specific classifiers |
| Application pages | 5 interactive pages |
| Explainability | SHAP |
| Class imbalance handling | SMOTE |
| Application framework | Streamlit |

## Live Application

[Launch the Streamlit Application](YOUR_STREAMLIT_APP_URL)

## Key Features

### 1. Multi-Disease Risk Prediction

Generates individual disease prediction probabilities using demographic, lifestyle and health-related information.

### 2. Overall Health Risk Score

Combines disease prediction outputs into an experimental overall health risk indicator.

### 3. Explainable Artificial Intelligence

Uses SHAP to provide global and individual prediction explanations, improving model transparency.

### 4. Exploratory Data Analysis Dashboard

Provides interactive visualisations to explore health indicators, disease distributions and dataset characteristics.

### 5. Model Leaderboard

Compares nine machine learning classifiers using accuracy, precision, recall, F1-score and ROC-AUC.

### 6. Health Recommendations

Provides general educational health information and preventive health guidance.

## Dataset

The project uses a prepared subset of the 2015 Behavioral Risk Factor Surveillance System (BRFSS) dataset.

The dataset contains 253,680 health survey records, including:

- Demographic information
- Body mass index
- Physical activity
- Smoking behaviour
- General health status
- Cholesterol indicators
- Existing health conditions
- Socioeconomic information

Original survey source: [CDC BRFSS 2015](https://www.cdc.gov/brfss/annual_data/annual_2015.html)

## Machine Learning Workflow

1. Dataset acquisition and understanding
2. Exploratory data analysis
3. Disease-specific feature selection
4. Training and testing data separation
5. Feature standardisation
6. Class imbalance handling using SMOTE
7. Machine learning model training
8. Model evaluation and comparison
9. Final model selection
10. SHAP-based prediction explanations
11. Streamlit application integration
12. System testing and verification

SMOTE was restricted to the training dataset to reduce the risk of data leakage during model evaluation.

## Machine Learning Algorithms

Three classification algorithms were implemented for each disease:

- Logistic Regression
- Random Forest
- Extreme Gradient Boosting (XGBoost)

A total of nine classification models were developed and evaluated.

## Model Performance

| Disease | Algorithm | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|---|---|---:|---:|---:|---:|---:|
| Diabetes | Logistic Regression | 73.31% | 0.82 | 0.31 | 0.76 | 0.44 |
| Diabetes | Random Forest | 85.01% | 0.79 | 0.44 | 0.28 | 0.34 |
| Diabetes | XGBoost | 86.41% | 0.82 | 0.53 | 0.20 | 0.29 |
| Heart Disease | Logistic Regression | 75.43% | 0.85 | 0.25 | 0.79 | 0.38 |
| Heart Disease | Random Forest | 89.32% | 0.81 | 0.38 | 0.22 | 0.28 |
| Heart Disease | XGBoost | 90.62% | 0.84 | 0.51 | 0.14 | 0.22 |
| High Blood Pressure | Logistic Regression | 72.30% | 0.80 | 0.66 | 0.73 | 0.69 |
| High Blood Pressure | Random Forest | 70.45% | 0.77 | 0.65 | 0.66 | 0.66 |
| High Blood Pressure | XGBoost | 73.18% | 0.80 | 0.69 | 0.69 | 0.69 |

*Evaluation metrics are rounded for presentation.*

### Selected Models

| Disease | Selected Model | Accuracy | ROC-AUC |
|---|---|---:|---:|
| Diabetes | XGBoost | 86.41% | 0.82 |
| Heart Disease | Logistic Regression | 75.43% | 0.85 |
| High Blood Pressure | XGBoost | 73.18% | 0.80 |

Model performance was assessed using multiple evaluation metrics rather than accuracy alone.

The selected diabetes model demonstrates relatively low positive-class recall, highlighting an important limitation for disease screening applications.

## Application Interface

The Streamlit application consists of five principal pages:

| Page | Description |
|---|---|
| Home | Project introduction and system overview |
| Risk Predictor | Individual disease prediction and risk assessment |
| EDA Dashboard | Dataset exploration and analytical visualisations |
| Model Leaderboard | Machine learning performance comparison |
| Health Recommendations | General health education and guidance |

## Technology Stack

### Programming and Development

- Python
- Google Colaboratory
- Git and GitHub

### Data Analysis and Visualisation

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost
- Imbalanced-learn
- SMOTE
- StandardScaler

### Explainable Artificial Intelligence

- SHAP

### Web Application

- Streamlit
- Pickle model serialisation

## Installation and Setup

Clone this repository:

`git clone YOUR_GITHUB_REPOSITORY_URL`

Navigate to the project directory:

`cd YOUR_REPOSITORY_NAME`

Create a virtual environment:

`python -m venv venv`

Activate the environment on Windows:

`venv\Scripts\activate`

Install the required dependencies:

`pip install -r requirements.txt`

Run the Streamlit application:

`streamlit run App/app.py`

## Explainable Artificial Intelligence

SHAP was incorporated to improve the interpretability of machine learning predictions through:

- Global feature importance
- Individual prediction explanations
- Feature contribution analysis
- Visual interpretation of model behaviour

These explanations describe the behaviour of the trained models and should not be interpreted as evidence of medical causation.

## Project Limitations

- The dataset contains historical, self-reported health survey information.
- The models have not been externally validated using Sri Lankan population data.
- Class imbalance continues to affect classification performance.
- The overall health risk score is experimental.
- The system has not undergone clinical validation.

## Future Improvements

- Integration of more recent healthcare datasets
- External model validation
- Hyperparameter and classification threshold optimisation
- Probability calibration
- Improved positive-class recall
- Formal usability evaluation
- Evidence-based health risk scoring

## Medical Disclaimer

This application was developed for academic, educational and research purposes only.

It is not a clinically validated diagnostic system and should not be used as a substitute for professional medical advice, diagnosis or treatment.

## Author

**Mohamed Faris Abdurrahman**

- BSc (Hons) Data Science – London Metropolitan University
- HND in Data Analytics
- Academic Project: 2026
- Areas of interest: Data Science, Machine Learning, Business Intelligence and Data Analytics

## Acknowledgements

Special thanks to the academic supervisors, lecturers and individuals who provided guidance and support throughout the development of this project.

---

Developed as an academic research project demonstrating the practical integration of machine learning, data analytics, explainable artificial intelligence and interactive web application development.
