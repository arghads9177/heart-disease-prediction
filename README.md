# Heart Disease Stage Prediction Project  

## Project Overview  
The **Heart Disease Stage Prediction Project** focuses on predicting the presence and stages of heart disease based on patient data. Using machine learning models and exploratory data analysis, this project aims to identify key factors contributing to heart disease, assist in early diagnosis, and provide actionable insights for healthcare providers.  

---

## Context  
This dataset is a **multivariate dataset**, meaning it involves various mathematical or statistical variables. It contains 14 primary attributes out of 76 available ones, which have been widely used in machine learning research.  
The **Cleveland database** is the most commonly utilized subset for heart disease prediction tasks.  

The main goals of this project are:  
1. To predict whether a person has heart disease based on given attributes.  
2. To analyze the dataset for insights that could improve understanding and early detection of heart disease.  

---

## About the Dataset  

### Column Descriptions  

| Column     | Description                                                                                       |
|------------|---------------------------------------------------------------------------------------------------|
| `id`       | Unique identifier for each patient.                                                              |
| `age`      | Age of the patient in years.                                                                      |
| `origin`   | Place of study where data was collected.                                                          |
| `sex`      | Gender of the patient (`Male`/`Female`).                                                          |
| `cp`       | Chest pain type (`typical angina`, `atypical angina`, `non-anginal`, `asymptomatic`).              |
| `trestbps` | Resting blood pressure (in mm Hg on admission to the hospital).                                   |
| `chol`     | Serum cholesterol level in mg/dl.                                                                 |
| `fbs`      | Fasting blood sugar (`True` if >120 mg/dl, else `False`).                                          |
| `restecg`  | Resting electrocardiographic results (`normal`, `st-t abnormality`, `lv hypertrophy`).            |
| `thalach`  | Maximum heart rate achieved during exercise.                                                      |
| `exang`    | Exercise-induced angina (`True`/`False`).                                                         |
| `oldpeak`  | ST depression induced by exercise relative to rest.                                               |
| `slope`    | Slope of the peak exercise ST segment.                                                            |
| `ca`       | Number of major vessels (0-3) colored by fluoroscopy.                                             |
| `thal`     | Results of the thalassemia test (`normal`, `fixed defect`, `reversible defect`).                  |
| `num`      | Predicted attribute (`0` = no heart disease; `1, 2, 3, 4` = stages of heart disease).             |

---

## Objectives  

1. **Predict Heart Disease Stages:**  
   - Classify patients into five stages (`0`, `1`, `2`, `3`, or `4`) based on clinical and test attributes.  

2. **Feature Importance:**  
   - Identify the most significant predictors of heart disease using statistical and machine learning models.  

3. **Risk Assessment:**  
   - Provide a decision-support tool for healthcare professionals to identify at-risk individuals.  

4. **Exploratory Data Analysis:**  
   - Uncover trends and correlations in the dataset to improve understanding of heart disease factors.  

---

## Methodology  

### 1. **Data Understanding**  
   - **Exploratory Data Analysis (EDA):** Perform statistical analysis and visualize data distributions, trends, and relationships.  
   - **Data Assessment:** Identify missing values, outliers, and inconsistencies.  

### 2. **Data Preparation**  
   - **Data Cleaning:** Handle missing values, remove duplicates, and normalize numerical attributes.  
   - **Feature Engineering:** Create derived features such as BMI, age group, and cholesterol categories.  
   - **Data Transformation:** Encode categorical variables and split data into training, validation, and test sets.  

### 3. **Modeling**  
   - **Baseline Models:** Use decision trees for initial benchmarks.  
   - **Advanced Models:** Train machine learning models such as Random Forest, XGBoost, and SVM.  
   - **Deep Learning:** Implement neural networks (e.g., MLP) for more complex patterns.  
   - **Hyperparameter Tuning:** Optimize models to enhance accuracy and efficiency.  

### 4. **Evaluation**  
   - Evaluate models using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and **ROC-AUC**.  
   - Perform cross-validation and compare models for robustness.  

---

## Tools and Technologies  

- **Programming Language:** Python  
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, TensorFlow/Keras  
- **Models:** Decision Tree, Random Forest, XGBoost, SVM, MLP  
- **Visualization Tools:** matplotlib, seaborn, plotly  

---

## Insights  

1. **Feature Significance:** Age, chest pain type, and maximum heart rate are critical predictors of heart disease.  
2. **Gender Differences:** Males have a slightly higher prevalence of heart disease in this dataset.  
3. **Risk Patterns:** Higher cholesterol levels and exercise-induced angina are linked to severe stages of heart disease.  

---

## Usage  

### Applications of the Project  

1. **Predictive Analytics:**  
   - Build models to predict heart disease stages for new patients.  

2. **Clinical Decision Support:**  
   - Provide tools for healthcare professionals to assess patient risk.  

3. **Healthcare Insights:**  
   - Identify high-risk groups and plan targeted interventions.  

4. **Educational Resource:**  
   - Serve as a dataset for students and researchers in healthcare data science.  

---

## Dataset Information  

- **Name:** Heart Disease Dataset  
- **Size:** Approximately 200 KB  
- **Format:** CSV  
- **Rows:** 920  
- **Columns:** 16

---

## Conclusion  

This project demonstrates the use of machine learning and data analysis techniques to predict heart disease stages. By leveraging patient data, it provides actionable insights for early diagnosis and effective management of heart disease.  

## Installation Guide

1. Intall python 3.11 in your system. from the following link https://www.python.org/downloads/
2. Install miniconda from the following link https://www.anaconda.com/download/success#miniconda
3. Open your Command Prompt or Terminal
4. Create the virtual environment `conda create -p venv`
5. Activate the virtual environment `conda activate venv\`
6. Install necessary packages `pip install -r UI\requirements.txt`
7. Move to the UI `cd UI`
7. Run the Application `streamlit run app.py`