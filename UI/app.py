import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


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

    # Sales Forecasting
    if choice == "Sales Forecasting":
        st.subheader("Future Sales Prediction")
        try:
            # Load the optimized Prophet model
            with open("../models/sf_prophet.pkl", "rb") as f:
                prophet_model = pickle.load(f)
            # Sidebar for user input
            st.header("Forecast Parameters")
            months_to_forecast = st.slider("Select the number of months for forecasting:", min_value=1, max_value=36, value=12)

            # Generate future dataframe
            st.header(f"Forecast for the Next {months_to_forecast} Months")
            last_date = prophet_model.history["ds"].max()
            future_dates = prophet_model.make_future_dataframe(periods=months_to_forecast, freq="M")
            forecast = prophet_model.predict(future_dates)

            # Filter the forecast for the selected range
            forecast_filtered = forecast[forecast["ds"] > last_date][["ds", "yhat", "yhat_lower", "yhat_upper"]]
            forecast_filtered.columns = ["Date", "Sales", "Lower", "Upper"]

            # Plot forecast
            st.subheader("Forecast Graph")
            fig, ax = plt.subplots(figsize=(10, 6))
            prophet_model.plot(forecast, ax=ax)
            st.pyplot(fig)

            # Display forecast values
            st.subheader("Forecasted Values")
            st.write(forecast_filtered.tail(months_to_forecast))

            # Additional option to download forecast
            st.subheader("Download Forecast")
            csv = forecast_filtered[["Date", "Sales", "Lower", "Upper"]].to_csv(index=False)
            st.download_button("Download Forecast as CSV", data=csv, file_name="forecast.csv", mime="text/csv")
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
        
    # Sub-Category Performance Analysis
    elif choice == "Sub-Category Performance Analysis":
        st.subheader("Sub-Category Performance Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "stores_sales_forecasting.csv")
        data = pd.read_csv(csv_path, encoding="latin1")
        # Find Sub-Category of customers
        col = "Sub-Category"
        subcat_values = data[col].value_counts().reset_index()
        st.markdown(f"### TOP 10 {col.upper()}(CUSTOMER COUNT) ")
        st.write(subcat_values.head(10))
        st.write(f"Total number of states: {subcat_values.shape[0]}")
        st.write("\n\n")

        # Visualize
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.barplot(data= subcat_values.head(10), x= col, y= "count", hue= col)
        plt.title(f"Top 10 {col} Counts")
        plt.xticks(rotation= 45)
        st.pyplot(fig)

        # Sub-Category Performance Analysis
        subcategory_performance = data.groupby("Sub-Category").agg(
            Total_Sales= ("Sales", "sum"),
            Total_Profit= ("Profit", "sum"),
            Avg_Sales= ("Sales", "mean"),
            Avg_Profit= ("Profit", "mean")
        ).reset_index()
        # Calculate Profit Percentage
        subcategory_performance["Profit_Percentage"] = subcategory_performance["Total_Profit"] / subcategory_performance["Total_Sales"] * 100

        st.markdown("### SALES AND PROFIT ANALYSIS FOR SUB-CATEGORY")
        st.write(subcategory_performance)
        st.write("\n\n")

        fig, ax = plt.subplots(figsize=(16, 6))
        # Visualize the Total Sales
        plt.subplot(1,2,1)
        sns.barplot(data = subcategory_performance, x= "Sub-Category", y= "Total_Sales", hue="Sub-Category")
        plt.title("Total Sales for Each Sub-Category")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=45)

        # Visualize the Profit Percentage
        plt.subplot(1,2,2)
        sns.barplot(data = subcategory_performance, x= "Sub-Category", y= "Profit_Percentage", hue="Sub-Category", palette="Set2")
        plt.title("Profit Percentage for Each Sub-Category")
        plt.ylabel("Profit(%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        st.markdown("""
        #### Top Performing Subcategory:

        - Furnishings have the highest profit margin (**14%**) and total profit (**$13,059**).

        #### Underperforming Subcategory:

        - Tables have a negative profit margin (**-8.56%**), indicating potential pricing or cost issues.

        #### Improvement Opportunities:

        - Focus on boosting sales of **Bookcases** and increasing margins for **Chairs**.
        - Investigate why **Tables** result in a loss and consider adjusting pricing or production costs.

        """)

        # Region wise Sub-Category Analysis
        region_subcategory_performance = data.groupby(["Region", "Sub-Category"]).agg(
            Total_Sales= ("Sales", "sum"),
            Total_Profit= ("Profit", "sum"),
            Avg_Sales= ("Sales", "mean"),
            Avg_Profit= ("Profit", "mean")
        ).reset_index()
        # Calculate Profit Percentage
        region_subcategory_performance["Profit_Percentage"] = region_subcategory_performance["Total_Profit"] / region_subcategory_performance["Total_Sales"] * 100

        st.markdown("### SALES AND PROFIT ANALYSIS FOR SUB-CATEGORY IN EACH REGION")
        st.write(region_subcategory_performance)
        st.write("\n\n")

        fig, ax = plt.subplots(figsize=(16, 6))
        # Visualize the Total Sales
        plt.subplot(1,2,1)
        sns.barplot(data = region_subcategory_performance, x= "Region", y= "Total_Sales", hue="Sub-Category")
        plt.title("Total Sales for Each Sub-Category Region Wise")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=45)

        # Visualize the Profit Percentage
        plt.subplot(1,2,2)
        sns.barplot(data = region_subcategory_performance, x= "Region", y= "Profit_Percentage", hue="Sub-Category", palette="Set2")
        plt.title("Profit Percentage for Each Sub-Category Region Wise")
        plt.ylabel("Profit(%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        # Key Findings
        st.markdown("""
        #### Top-performing regions and subcategories:

        - **Chairs** have highest sales in **West** region and a very decent sale in **East and Central** regions as well.
        - **Furnishings** have the highest profit margin(**25%**) in **West** region and maintain a decent profit margin(**20%**) in **East and South** regions.

        #### Underperforming products in specific regions:

        - **Tables** in the **East** region show a negative profit margin (**-28%**), requiring investigation into pricing or cost-related issues.

        #### Improvement Opportunities:

        - Focus on boosting sales of **Bookcases** in **South** region and increasing margins for **Chairs** in **West** Region.
        - Investigate why **Furnishings** in **Central** region result in a loss and consider adjusting pricing or production costs.
        - Investigate why **Tables** in all regions except **West** region result in a loss and consider adjusting pricing or production costs.
        """)
    # Product Performance Analysis
    elif choice == "Product Performance Analysis":
        st.subheader("Product Performance Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "stores_sales_forecasting.csv")
        data = pd.read_csv(csv_path, encoding="latin1")

        # Product Performance Analysis
        # Calculate Total, Sales, Proft, Quantity Sold and Rank the accrodingly
        product_performance = data.groupby("Product Name").agg(
            Total_Sales= ("Sales", "sum"),
            Total_Profit= ("Profit", "sum"),
            Avg_Sales= ("Sales", "mean"),
            Avg_Profit= ("Profit", "mean"),
            Total_Quantity= ("Quantity", "sum")
        ).reset_index()
        # Calculate Profit Percentage
        product_performance["Profit_Percentage"] = product_performance["Total_Profit"] / product_performance["Total_Sales"] * 100

        # Set Rank of Products by Total Sales, Quantity and Profut
        product_performance["Sales_Rank"] = product_performance["Total_Sales"].rank(ascending= False)
        product_performance["Profit_Rank"] = product_performance["Profit_Percentage"].rank(ascending= False)
        product_performance["Quantity_Rank"] = product_performance["Total_Quantity"].rank(ascending= False)

        st.markdown("### TOP 10 HIGH SELLING PRODUCTS")
        st.write(product_performance[["Product Name", "Total_Sales", "Avg_Sales", "Sales_Rank"]].sort_values("Sales_Rank").head(10))
        st.write("\n\n")

        fig, ax = plt.subplots(figsize=(16, 6))
        # Visualize the Total Sales
        sns.barplot(data = product_performance.sort_values("Sales_Rank").head(10), x= "Product Name", y= "Total_Sales", hue="Product Name")
        plt.title("Top 10 Selling Products")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown("### TOP 10 HIGH PROFIT PRODUCTS")
        st.write(product_performance[["Product Name", "Total_Profit", "Profit_Percentage", "Profit_Rank"]].sort_values("Profit_Rank").head(10))
        st.write("\n\n")

        # Visualize the Profit Percentage
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(data = product_performance.sort_values("Profit_Rank").head(10), x= "Product Name", y= "Profit_Percentage", hue="Product Name")
        plt.title("Top 10 Profitable Products")
        plt.ylabel("Profit (%)")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown("### TOP 10 HIGH DEMAND PRODUCTS")
        st.write(product_performance[["Product Name", "Total_Quantity", "Quantity_Rank"]].sort_values("Quantity_Rank").head(10))
        st.write("\n\n")

        # Visualize the Low Demand
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(data = product_performance.sort_values("Quantity_Rank").head(10), x= "Product Name", y= "Total_Quantity", hue="Product Name")
        plt.title("Top 10 Demanding Products")
        plt.ylabel("Total Quantity")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.markdown("### 10 LOW DEMAND PRODUCTS")
        st.write(product_performance[["Product Name", "Total_Quantity", "Quantity_Rank"]].sort_values("Quantity_Rank").tail(10))
        st.write("\n\n")

        # Visualize the Low Demand
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.barplot(data = product_performance.sort_values("Quantity_Rank").tail(10), x= "Product Name", y= "Total_Quantity", hue="Product Name")
        plt.title("10 Low Demanding Products")
        plt.ylabel("Total Quantity")
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # Categorize the Products in terms of Sales and Profit Margin
        # Define thresholds (using median as an example)
        sales_median = product_performance['Total_Sales'].median()
        profit_median = product_performance['Profit_Percentage'].median()

        # Categorize products
        def categorize(row):
            if row['Total_Sales'] > sales_median and row['Profit_Percentage'] > profit_median:
                return 'High Sales, High Profit'
            elif row['Total_Sales'] > sales_median and row['Profit_Percentage'] <= profit_median:
                return 'High Sales, Low Profit'
            elif row['Total_Sales'] <= sales_median and row['Profit_Percentage'] > profit_median:
                return 'Low Sales, High Profit'
            else:
                return 'Low Sales, Low Profit'

        product_performance['Segment'] = product_performance.apply(categorize, axis=1)

        # Count of products in each segment
        st.markdown("### SEGMENTATION SUMMARY")
        segment_counts = product_performance['Segment'].value_counts()
        st.write(segment_counts)

        # Visualization: Scatter plot of Sales vs. Profit
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.scatterplot(
            data=product_performance,
            x='Total_Sales', y='Profit_Percentage',
            hue='Segment', palette='Set2'
        )
        plt.title('Product Segmentation: Sales vs. Profit')
        plt.xlabel('Total Sales')
        plt.ylabel('Profit (%)')
        plt.legend(title='Segment')
        st.pyplot(fig)

        st.markdown("""
        #### High Performers:

        - Products with high sales and profit margin should be prioritized for promotions or further analysis to replicate their success.

        #### Low Performers:

        - Products with low sales and profit margins should be reviewed for potential issues (e.g., high costs, low demand).
        #### Opportunities:

        - Products with high sales but low profit margins may require price optimization or cost reduction strategies.

        #### Segmentation

        - **High Sales, High Profit:**

            - These are your star products. Focus on expanding their reach or replicating their success.

        - **High Sales, Low Profit:**

            - These are potential opportunities for cost optimization or price adjustment to improve profitability.

        - **Low Sales, High Profit:**

            - These products are niche but profitable. Consider targeted marketing to grow their sales.

        - **Low Sales, Low Profit:**

            - These are underperforming products. Investigate for possible discontinuation, rebranding, or cost reduction.
        """)
        # Market Basket Analysis
    elif choice == "Market Basket Analysis":
        st.subheader("Market Basket Analysis")
        data_path = "../data"
        csv_path = os.path.join(data_path, "stores_sales_forecasting.csv")
        data = pd.read_csv(csv_path, encoding="latin1")
        # Prepare Data: Convert the data to basket format
        basket= data.groupby(["Order ID", "Sub-Category"])["Quantity"].sum().unstack().fillna(0)
        basket= basket.applymap(lambda x: 1 if x > 0 else 0)
        # Find frequent itemsets with a minimum support threshold
        frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
        # Generate Association rules
        rules = association_rules(frequent_itemsets, metric= "lift", min_threshold= 0.3)

        # Visualize top 10 rules by Lift
        top_rules = rules.head(10)
        fig, ax = plt.subplots(figsize=(16, 6))
        plt.barh(range(len(top_rules)), top_rules['lift'], color='skyblue')
        plt.yticks(range(len(top_rules)), [f"{list(a)} -> {list(c)}" for a, c in zip(top_rules['antecedents'], top_rules['consequents'])])
        plt.xlabel('Lift')
        plt.ylabel('Association Rules')
        plt.title('Top 10 Association Rules by Lift')
        st.pyplot(fig)

        # key Findings
        st.markdown("""
        From the given chart showing the **Top 10 Association Rules by Lift**, we can derive the following insights:

        #### Frequent Co-Purchases:

        - Customers frequently buy Chairs and Tables together. This indicates a strong complementary relationship between these products.
        - Similarly, Furnishings and Chairs are often bought in combination, suggesting they are likely used in the same setting or context (e.g., home/office decor).

        #### Bidirectional Relationships:

        The rules highlight bidirectional associations. For instance:
        - ["Chairs"] → ["Tables"]
        - ["Tables"] → ["Chairs"] These reciprocal rules indicate strong dependencies, meaning customers often consider both items when making purchasing decisions.

        #### Lift Values:

        - Lift values for these associations are relatively **low (below 1)**, which suggests these rules, while valid, may not represent highly dominant patterns across the entire dataset.
        - A lower lift could mean that while the rules are valid, **the combinations are not highly exceptional compared to random co-purchases**.

        #### Potential Cross-Selling Opportunities:

        - **Furnishings and Chairs** have an association. Stores can bundle these items or place them in close proximity to encourage cross-sales.
        - The relationship between **Chairs and Tables** suggests a similar bundling or discount strategy.

        #### Segmentation-Based Promotions:

        - The rules point to specific product categories that tend to be purchased together. Marketing campaigns targeting customers purchasing Chairs could include discounts or recommendations for Tables or Furnishings.

        #### Customer Behavior Insight:

        - These associations reveal customers' preference to purchase items that complement each other in functionality or aesthetics, particularly for home or office use.
        """)

#Calling the main function
if __name__ == "__main__":
    main()