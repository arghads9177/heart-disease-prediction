import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle


# Define the prediction function
def predict_heart_disease(model, features):
    return model.predict([features])[0]

def link_style_menu():
    st.sidebar.markdown(
        """
        <style>
        .sidebar-link {
            font-size: 18px;
            color: #007BFF;
            text-decoration: none;
            margin-bottom: 10px;
            display: block;
        }
        .sidebar-link:hover {
            text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    menu = {
        "Heart Disease Prediction": "Heart Disease Prediction",
        "Target Analysis": "Target Analysis",
        "Demographic Analysis": "Demographic Analysis",
        "Clinical Analysis": "Clinical Analysis"
    }
    
    # Display link-style menu
    selected_option = st.sidebar.radio(
        "",
        options=list(menu.values()),
        label_visibility="collapsed"
    )
    
    return selected_option

# Streamlit app
def main():
    st.title("Heart Disease Prediction and Analysis")
    choice = link_style_menu()

    # Heart Disease Prediction
    if choice == "Heart Disease Prediction":
        st.subheader("Heart Disease Prediction")
        try:
            # Load the optimized Prophet model
            with open("../models/model_xgb.pkl", "rb") as f:
                model = pickle.load(f)
            
            st.write("Predict the stage of heart disease based on input features.")

            # User input fields
            age = st.number_input("Age of the patient in years:", min_value=1, max_value=120, step=1)
            sex = st.selectbox("Gender of the patient:", options=["Male", "Female"])
            cp = st.selectbox("Chest pain type:", options=["Typical angina", "Atypical angina", "Non-anginal", "Asymptomatic"])
            trestbps = st.number_input("Resting blood pressure (mm Hg):", min_value=50, max_value=250, step=1)
            chol = st.number_input("Serum cholesterol level (mg/dl):", min_value=50, max_value=600, step=1)
            fbs = st.radio("Fasting blood sugar >120 mg/dl:", options=["True", "False"])
            restecg = st.selectbox("Resting electrocardiographic results:", options=["Normal", "ST-T abnormality", "LV hypertrophy"])
            thalach = st.number_input("Maximum heart rate achieved during exercise:", min_value=50, max_value=220, step=1)
            exang = st.radio("Exercise-induced angina:", options=["True", "False"])
            oldpeak = st.number_input("ST depression induced by exercise relative to rest:", min_value=0.0, max_value=10.0, step=0.1)
            slope = st.selectbox("Slope of the peak exercise ST segment:", options=["Upsloping", "Flat", "Downsloping"])
            ca = st.slider("Number of major vessels colored by fluoroscopy:", min_value=0, max_value=3, step=1)
            thal = st.selectbox("Thalassemia test result:", options=["Normal", "Fixed defect", "Reversible defect"])

            # Encode categorical variables
            sex_encoded = 1 if sex == "Male" else 0
            fbs_encoded = 1 if fbs == "True" else 0
            exang_encoded = 1 if exang == "True" else 0
            cp_encoded = ["Asymptomatic", "Typical angina", "Atypical angina", "Non-anginal"].index(cp) - 1
            restecg_encoded = ["Normal", "LV hypertrophy", "ST-T abnormality"].index(restecg)
            slope_encoded = ["Upsloping", "Flat", "Downsloping"].index(slope)
            thal_encoded = ["Normal", "Fixed defect", "Reversible defect"].index(thal)

            # Combine features into a single array
            features = [
                age, 
                sex_encoded, 
                cp_encoded, 
                trestbps, 
                chol, 
                fbs_encoded, 
                restecg_encoded, 
                thalach, 
                exang_encoded, 
                oldpeak, 
                slope_encoded, 
                ca, 
                thal_encoded
            ]

            # Predict button
            if st.button("Predict Heart Disease Stage"):
                result = predict_heart_disease(model, features)
                # Define a mapping for severity levels
                severity_mapping = {
                    0: "Normal",
                    1: "Mild",
                    2: "Moderate",
                    3: "Severe",
                    4: "Very Severe"
                }
                severity = severity_mapping.get(result, "Unknown")
                st.write(f"Predicted Heart Disease Stage: **{severity}**")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Target Analysis
    elif choice == "Target Analysis":
        st.subheader("Target Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "hd_uci_cleaned.csv")
        df = pd.read_csv(csv_path)
        # Check class imbalance
        num_percentage = df["num"].value_counts(normalize= True) * 100
        st.write("Percentage Distribution of Target Variable:")
        st.write(num_percentage)

        # Visualize Distribution of target
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.countplot(data= df, x= "num", hue= "num")
        plt.title("Distribution of Heart Disease Stages")
        plt.xlabel("Stages")
        plt.ylabel("Count")
        st.pyplot(fig)
        # Convert to binary classification for simpler analysis
        df['binary_target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
        binary_counts = df['binary_target'].value_counts(normalize=True) * 100
        st.write("\nPercentage Distribution for Binary Target:")
        st.write(binary_counts)

        # Visualize Distribution of target(binary classification)
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.countplot(data= df, x= "binary_target", hue= "binary_target")
        plt.title("Distribution of Binary Classification of Heart Disease")
        plt.xlabel("Heart Disease")
        plt.ylabel("Count")
        st.pyplot(fig)
        # Title
        st.markdown("""
        ### Target Variable Analysis Insights

        #### **Percentage Distribution of Target Variable (`num`):**
        - The majority of patients fall into **`num=0` (No Heart Disease)**, accounting for **44.67%** of the dataset.
        - Among patients with heart disease stages (`num=1, 2, 3, 4`):
        - **`num=1`** is the most prevalent stage, comprising **28.80%** of the dataset.
        - **`num=2`** and **`num=3`** are relatively balanced, with **11.85%** and **11.63%**, respectively.
        - **`num=4`** (severe heart disease) is the least common, with only **3.04%** of the dataset.

        #### **Percentage Distribution for Binary Target (`binary_target`):**
        - When converting the target into a binary classification:
        - **`binary_target=1` (Heart Disease)** accounts for **55.33%** of the dataset.
        - **`binary_target=0` (No Heart Disease)** accounts for **44.67%** of the dataset.
        - The binary target shows a relatively balanced distribution with a slight majority of cases having heart disease.

        #### **Key Observations:**
        - The dataset exhibits a moderate class imbalance in the multi-class target (`num`), with **`num=4`** being underrepresented.
        - The binary target (`binary_target`) is more balanced and suitable for binary classification tasks, making it an attractive option for predictive modeling.
        - The prevalence of `num=1` highlights that mild heart disease is the most common stage among patients with heart conditions.

        #### **Implications for Analysis and Modeling:**
        - The class imbalance in `num` should be addressed using techniques such as SMOTE or class weighting during model training.
        - Binary classification models might yield better generalization due to the balanced distribution of `binary_target`.
        - Further analysis should explore feature relationships with both the multi-class and binary targets to understand patterns in heart disease progression.
        """)
    # Demographic Analysis
    elif choice == "Demographic Analysis":
        st.subheader("Demographic Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "hd_uci_cleaned.csv")
        df = pd.read_csv(csv_path)
        st.markdown("""
        ### Age Analysis

        Anlyze the distribution of age and how stages of heart disease is influenced with different ages of patients.   
        """)
        # Find Statistical Info of Age
        col= "age"
        age_stats = df[col].describe()
        st.write(f"Age STATISTICS")
        st.write(age_stats)
        # Visualize
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.histplot(data= df, x= col, kde= True)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.boxplot(x='num', y=col, data=df, palette='viridis')
        plt.title(f'{col} Distribution Across Heart Disease Stages')
        plt.xlabel('Heart Disease Stage')
        plt.ylabel(col)
        st.pyplot(fig)

        st.markdown("""
        ### Age Analysis Insights

        #### **Descriptive Statistics of Age:**
        - **Count:** The dataset includes 920 observations for the `age` feature.
        - **Mean Age:** The average age is approximately **53.51 years**, indicating that the dataset predominantly represents middle-aged individuals.
        - **Standard Deviation:** The standard deviation is **9.42 years**, suggesting a moderate spread in the ages.
        - **Minimum Age:** The youngest individual in the dataset is **28 years old**.
        - **Maximum Age:** The oldest individual in the dataset is **77 years old**.
        - **Quartiles:**
        - **25th Percentile:** 25% of the individuals are younger than **47 years**.
        - **50th Percentile (Median):** The median age is **54 years**.
        - **75th Percentile:** 75% of the individuals are younger than **60 years**.

        #### **Insights from the Box Plot:**
        - The age distribution shows a clear trend across heart disease stages:
        - Individuals in **stage 0 (No Heart Disease)** have a slightly wider age range, with younger patients included.
        - The median age increases with higher heart disease stages, suggesting that older individuals are more likely to develop advanced stages of heart disease.
        - **Stage 4 (Severe Heart Disease)** has the smallest interquartile range, indicating that the affected individuals are concentrated within a narrower age group.

        #### **Key Observations:**
        - Age is a significant factor associated with heart disease progression. The likelihood of severe heart disease increases with age.
        - Patients in **stage 0** (No Heart Disease) tend to have a broader age distribution, potentially reflecting a mix of younger and middle-aged individuals with healthier cardiovascular profiles.
        - Outliers in the higher stages (e.g., younger individuals with stage 3 or stage 4) may indicate unique cases of early-onset heart disease due to genetic or lifestyle factors.

        #### **Implications:**
        - Age should be considered a critical variable in predictive modeling for heart disease risk.
        - Preventive measures should focus on middle-aged and older populations to mitigate heart disease progression.
        - Further analysis could explore the interaction between age and other risk factors (e.g., cholesterol, blood pressure) to identify high-risk profiles.

        """)
        st.markdown("""
        ### Gender Analysis

        Analyze how different stages of heart disease affects the male and female.
        """)
        # Count target by gender
        col = "sex"
        gender_count = df.groupby(["num", col]).size().unstack()
        st.write("Gender Analysis")
        st.write(gender_count)
        st.write("\n\n")

        # Visualize
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.countplot(data= df, x= "num", hue= col)
        plt.title("Heart Disease in male and female")
        st.pyplot(fig)

        st.markdown("""
        ### Gender Analysis Insights

        - **Stage 0 (No Heart Disease):**
        - **Female:** 144 patients
        - **Male:** 267 patients
        - **Insight:** Males represent a significantly higher proportion (65%) of patients with no heart disease compared to females (35%).

        - **Stage 1 (Mild Heart Disease):**
        - **Female:** 30 patients
        - **Male:** 235 patients
        - **Insight:** Males dominate this stage with a significant majority (89%), while females constitute only 11%.

        - **Stage 2 (Moderate Heart Disease):**
        - **Female:** 10 patients
        - **Male:** 99 patients
        - **Insight:** Males are over 9 times more likely to be in Stage 2 than females, indicating a strong gender disparity in moderate heart disease.

        - **Stage 3 (Severe Heart Disease):**
        - **Female:** 8 patients
        - **Male:** 99 patients
        - **Insight:** Males account for the vast majority (93%) of patients in this stage, with females making up only 7%.

        - **Stage 4 (Critical Heart Disease):**
        - **Female:** 2 patients
        - **Male:** 26 patients
        - **Insight:** Males comprise 93% of critical heart disease cases, while females represent only 7%, indicating a highly skewed gender distribution at this stage.

        ### Key Observations:
        - Across all stages of heart disease, males consistently outnumber females.
        - The disparity becomes more pronounced in advanced stages (Stages 2, 3, and 4), where males overwhelmingly represent the majority of patients.
        - This suggests that males may either have a higher predisposition to developing heart disease or are more likely to progress to advanced stages than females. Further analysis is needed to explore the underlying causes, such as lifestyle factors, genetic predisposition, or healthcare-seeking behavior.

        """)
        
    # Clinical Analysis
    elif choice == "Clinical Analysis":
        st.subheader("Clinical Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "hd_uci_cleaned.csv")
        df = pd.read_csv(csv_path)
        st.markdown("""
        ### Resting Blood Pressure (trestbps) Analysis

        Analyze the distribution of resting blood pressure and its relationship with heart disease progression.
        """)
        # Find Statistical Info of BP
        col= "trestbps"
        bp_stats = df[col].describe()
        st.write(f"{col.upper()} STATISTICS")
        st.write(bp_stats)
        st.write("\n\n")

        # Visualize Distribution of Age
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.histplot(data= df, x= col, kde= True)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

        # Relationship of BP with target
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='num', y=col, data=df, palette='viridis')
        plt.title(f'{col} Distribution Across Heart Disease Stages')
        plt.xlabel('Heart Disease Stage')
        plt.ylabel(col)
        st.pyplot(fig)

        st.markdown("""
        ### Resting Blood Pressure (trestbps) Analysis

        #### Statistical Summary:
        - **Count:** 920 observations
        - **Mean:** 131.99 mm Hg
        - **Standard Deviation:** 18.45 mm Hg
        - **Minimum:** 0 mm Hg (likely erroneous or missing value)
        - **25th Percentile:** 120 mm Hg
        - **Median (50th Percentile):** 130 mm Hg
        - **75th Percentile:** 140 mm Hg
        - **Maximum:** 200 mm Hg

        #### Observations from the Boxplot:
        - **Stage 0 (No Heart Disease):**
        - Resting blood pressure is relatively stable, with the majority of values clustered around the median (~130 mm Hg). 
        - Few outliers below the 100 mm Hg mark suggest potential erroneous or rare data points.

        - **Stages 1 to 4 (Heart Disease Stages):**
        - An increasing trend in median resting blood pressure is observed with advanced heart disease stages.
        - The interquartile range (IQR) widens slightly in later stages, indicating increased variability in resting blood pressure levels.
        - Outliers above 175 mm Hg are present across all stages, but their frequency increases in advanced stages.

        #### Key Insights:
        1. **Progression Trend:**
        - Patients with more advanced heart disease stages tend to have higher median resting blood pressure levels, suggesting a potential correlation between elevated blood pressure and disease severity.

        2. **Data Quality Concerns:**
        - The presence of a minimum value of 0 mm Hg likely indicates a data entry error or missing value, requiring further investigation or imputation.

        3. **Clinical Implications:**
        - High resting blood pressure levels could serve as an important indicator for assessing heart disease risk and its progression.
        - Consistent monitoring of resting blood pressure is critical for early detection and management of heart disease.

        4. **Outliers:**
        - Extreme values, both high and low, could indicate additional underlying conditions or measurement inaccuracies.

        #### Recommendations:
        - Address the erroneous values (e.g., 0 mm Hg) through imputation or exclusion.
        - Further analysis could investigate the relationship between resting blood pressure and other key features such as cholesterol, age, and maximum heart rate.

        """)

        st.markdown("""
        ### Cholesterol Analysis

        Analyze the distribution of cholesterol levels and its relationship with heart disease progression.
        """)
        # Find Statistical Info of cholestoral
        col= "chol"
        chol_stats = df[col].describe()
        st.write(f"{col.upper()} STATISTICS")
        st.write(chol_stats)

        # Visualize Distribution of Cholestoral
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data= df, x= col, kde= True)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

        # Relationship of cholestoral with target
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='num', y=col, data=df, palette='viridis')
        plt.title(f'{col} Distribution Across Heart Disease Stages')
        plt.xlabel('Heart Disease Stage')
        plt.ylabel(col)
        st.pyplot(fig)

        st.markdown("""
        ### Cholesterol (chol) Analysis

        #### Statistical Summary:
        - **Count:** 919 observations
        - **Mean:** 200.13 mg/dl
        - **Standard Deviation:** 108.90 mg/dl
        - **Minimum:** 0 mg/dl (indicates missing or erroneous entries, found in 171 rows)
        - **25th Percentile:** 178.5 mg/dl
        - **Median (50th Percentile):** 223 mg/dl
        - **75th Percentile:** 267 mg/dl
        - **Maximum:** 603 mg/dl

        #### Observations from the Boxplot:
        - **Stage 0 (No Heart Disease):**
        - Median cholesterol level is slightly lower compared to patients with heart disease stages.
        - Several outliers are observed above 400 mg/dl, indicating extreme cholesterol levels.

        - **Stages 1 to 4 (Heart Disease Stages):**
        - Median cholesterol levels increase slightly with advanced heart disease stages.
        - The interquartile range (IQR) expands, particularly in stages 2 and 3, suggesting greater variability in cholesterol levels among these groups.
        - Extreme outliers above 500 mg/dl are present across all stages.

        #### Key Insights:
        1. **Erroneous Data:**
        - The presence of 171 rows with cholesterol levels of 0 indicates missing or incorrect data. These need to be imputed or excluded to ensure accurate analysis.

        2. **Correlation with Heart Disease Stages:**
        - Cholesterol levels tend to increase with advanced stages of heart disease, highlighting its importance as a risk factor.
        - The higher median and broader range in advanced stages suggest that cholesterol levels are a crucial indicator for disease severity.

        3. **Outliers:**
        - Extremely high cholesterol levels (above 400 mg/dl) are present in all stages and should be further examined to understand their clinical significance.

        4. **Clinical Implications:**
        - High cholesterol is a well-established risk factor for heart disease. Patients with elevated levels should consider lifestyle changes and medical interventions to mitigate risks.

        #### Recommendations:
        - Impute or exclude rows with cholesterol levels of 0 to ensure the validity of further analyses.
        - Investigate the relationship between cholesterol and other factors such as age, resting blood pressure, and heart disease stage.
        - Use cholesterol levels as a key predictor in machine learning models for heart disease risk prediction.

        #### Boxplot Interpretation:
        - The attached chart effectively visualizes the distribution of cholesterol levels across different stages of heart disease. It highlights the increasing trend and variability with disease progression.
        """)
        

#Calling the main function
if __name__ == "__main__":
    main()