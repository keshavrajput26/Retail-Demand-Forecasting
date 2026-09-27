# Retail Sales & Demand Forecasting — Complete Project Guide

## 0. Project Goal

Build an end-to-end data science project for an **Associate Data Scientist** application.

### Business problem

Retailers need to estimate sales to support inventory planning, promotions, and operational decisions.

### Project objective

Analyze historical retail store sales and build machine-learning models to predict daily sales using:

- Store characteristics
- Promotions
- Holidays
- Day of week
- Month/year
- Competition-related information

### Core workflow

```text
Problem Definition
        ↓
Dataset Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
SQL Analysis
        ↓
Chronological Train/Test Split
        ↓
Linear Regression Baseline
        ↓
Random Forest Regressor
        ↓
Model Evaluation
        ↓
Feature Importance
        ↓
Business Insights
        ↓
GitHub + Resume
```

---

# 1. Why This Project?

This project demonstrates:

- Statistics and descriptive analysis
- Data cleaning
- Exploratory data analysis
- Python/pandas/NumPy
- Data visualization
- Feature engineering
- SQL
- Regression
- Model evaluation
- Business interpretation
- Git/GitHub portfolio skills

The project should be presented as an **end-to-end business data science case study**, not simply as a model-training exercise.

---

# 2. Dataset

Use the **Rossmann Store Sales** dataset.

Expected files:

```text
train.csv
store.csv
```

The main target variable is:

```text
Sales
```

Important columns include:

| Column | Meaning |
|---|---|
| Store | Store ID |
| DayOfWeek | Day of week |
| Date | Date |
| Sales | Daily sales — target |
| Customers | Number of customers |
| Open | Whether the store was open |
| Promo | Whether a promotion was running |
| StateHoliday | State holiday indicator |
| SchoolHoliday | School holiday indicator |

`store.csv` contains additional store information such as:

- Store type
- Assortment
- Competition distance
- Competition opening information
- Promo2 information

---

# 3. Project Folder Structure

Create:

```text
retail-demand-forecasting/
│
├── data/
│   ├── train.csv
│   └── store.csv
│
├── notebooks/
│   └── retail_demand_forecasting.ipynb
│
├── sql/
│   └── analysis.sql
│
├── images/
│   ├── sales_distribution.png
│   ├── sales_over_time.png
│   ├── sales_by_day.png
│   ├── promotion_analysis.png
│   └── actual_vs_predicted.png
│
├── README.md
├── PROJECT_GUIDE.md
└── requirements.txt
```

Do not upload large raw datasets to GitHub unless appropriate. Put dataset acquisition instructions in the README instead.

---

# 4. Notebook Structure

The notebook should contain:

1. Problem Definition
2. Imports
3. Load Data
4. Data Understanding
5. Data Cleaning
6. Merge Datasets
7. Exploratory Data Analysis
8. Feature Engineering
9. Feature Selection
10. Encoding
11. Train/Test Split
12. Linear Regression
13. Random Forest
14. Model Evaluation
15. Actual vs Predicted
16. Feature Importance
17. Business Insights
18. Limitations
19. Future Improvements
20. Optional Store Clustering

---

# 5. Section 1 — Problem Definition

Use a Markdown cell:

```markdown
# Retail Sales & Demand Forecasting

## Business Problem

Retail businesses need to estimate future sales to support inventory
planning, promotional decisions, and operational planning.

## Objective

The objective of this project is to analyze historical retail sales data
and develop machine learning models to predict daily store sales.

## Approach

The project follows an end-to-end data science workflow:

1. Data understanding
2. Data cleaning
3. Exploratory data analysis
4. Feature engineering
5. SQL-based analysis
6. Machine learning
7. Model evaluation
8. Business interpretation
```

---

# 6. Section 2 — Import Libraries

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

pd.set_option("display.max_columns", None)
```

---

# 7. Section 3 — Load Data

If the notebook is inside `notebooks/`:

```python
train = pd.read_csv("../data/train.csv")
store = pd.read_csv("../data/store.csv")
```

If that path does not work, adjust it according to your working directory.

Inspect:

```python
train.head()
```

```python
store.head()
```

```python
print("Train shape:", train.shape)
print("Store shape:", store.shape)
```

```python
train.info()
```

```python
store.info()
```

---

# 8. Section 4 — Understand the Data

Check descriptive statistics:

```python
train.describe()
```

Check missing values:

```python
train.isnull().sum()
```

```python
store.isnull().sum()
```

Check the target:

```python
train["Sales"].describe()
```

Check duplicates:

```python
train.duplicated().sum()
```

```python
store.duplicated().sum()
```

### Questions to answer

- How many rows are there?
- How many columns?
- What is the target variable?
- Which columns are numerical?
- Which are categorical?
- Are there missing values?
- Are there duplicates?
- What is the range of sales?

---

# 9. Section 5 — Merge the Datasets

The sales data and store metadata are separate.

Merge using `Store`:

```python
df = train.merge(
    store,
    on="Store",
    how="left"
)
```

Inspect:

```python
df.head()
```

```python
df.shape
```

```python
df.info()
```

Check whether the merge introduced missing values:

```python
df.isnull().sum()
```

---

# 10. Section 6 — Handle Closed Stores

Check:

```python
df["Open"].value_counts()
```

For the initial sales-prediction problem, focus on open stores:

```python
df = df[df["Open"] == 1].copy()
```

Then:

```python
df.shape
```

```python
df["Sales"].describe()
```

### Why?

Closed stores have zero sales because they were not operating. The initial objective is to predict sales for operating stores, so keeping the analysis focused on open stores makes the target more meaningful.

---

# 11. Section 7 — Date Processing

Check the date type:

```python
df["Date"].dtype
```

Convert it:

```python
df["Date"] = pd.to_datetime(df["Date"])
```

Create date features:

```python
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
```

Inspect:

```python
df[
    [
        "Date",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "WeekOfYear"
    ]
].head()
```

### Important concept

Feature engineering means transforming raw data into useful variables for analysis or machine learning.

For example:

```text
2015-07-31
      ↓
Year = 2015
Month = 7
Day = 31
DayOfWeek = 4
WeekOfYear = 31
```

---

# 12. Section 8 — Exploratory Data Analysis

Do not create dozens of random charts.

Aim for around 5–6 meaningful visualizations.

---

## 12.1 Sales Distribution

```python
plt.figure(figsize=(10, 5))

sns.histplot(df["Sales"], bins=50)

plt.title("Distribution of Daily Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.show()
```

### Interpretation

Look for:

- Skewness
- Typical sales range
- Very high-sales observations
- Spread of the target

Do not claim that unusual observations are errors without investigating them.

---

# 13. Sales Over Time

Aggregate sales by date:

```python
daily_sales = df.groupby("Date")["Sales"].sum()
```

Plot:

```python
plt.figure(figsize=(14, 5))

plt.plot(daily_sales)

plt.title("Total Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")

plt.show()
```

### Questions

Look for:

- Trends
- Seasonal patterns
- Peaks
- Dips
- Unusual periods

---

# 14. Sales by Day of Week

```python
day_sales = df.groupby("DayOfWeek")["Sales"].mean()
```

```python
plt.figure(figsize=(8, 5))

day_sales.plot(kind="bar")

plt.title("Average Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")

plt.show()
```

Remember:

```text
0 = Monday
1 = Tuesday
2 = Wednesday
3 = Thursday
4 = Friday
5 = Saturday
6 = Sunday
```

---

# 15. Promotion vs Sales

Calculate:

```python
promo_sales = df.groupby("Promo")["Sales"].mean()

print(promo_sales)
```

Plot:

```python
plt.figure(figsize=(7, 5))

promo_sales.plot(kind="bar")

plt.title("Average Sales by Promotion Status")
plt.xlabel("Promotion")
plt.ylabel("Average Sales")

plt.show()
```

### Interpretation

Compare observed average sales between promotional and non-promotional periods.

Avoid claiming causation.

Better:

> Average observed sales differed between promotional and non-promotional periods.

Avoid:

> Promotions caused sales to increase by X%.

unless a proper causal analysis has been performed.

---

# 16. Sales by Store Type

```python
store_type_sales = df.groupby("StoreType")["Sales"].mean()

print(store_type_sales)
```

```python
plt.figure(figsize=(8, 5))

store_type_sales.plot(kind="bar")

plt.title("Average Sales by Store Type")
plt.xlabel("Store Type")
plt.ylabel("Average Sales")

plt.show()
```

---

# 17. Monthly Sales

```python
monthly_sales = df.groupby("Month")["Sales"].mean()
```

```python
plt.figure(figsize=(10, 5))

monthly_sales.plot(kind="bar")

plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.show()
```

Use this to discuss potential seasonal patterns.

---

# 18. Correlation Analysis

Select numerical variables:

```python
numeric_cols = [
    "Sales",
    "Customers",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "Year",
    "Month",
    "DayOfWeek"
]
```

Create a correlation matrix:

```python
plt.figure(figsize=(10, 7))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()
```

### Important theory

Correlation measures the strength and direction of a linear relationship.

Roughly:

```text
+1 → strong positive linear relationship
 0 → little linear relationship
-1 → strong negative linear relationship
```

Correlation does not prove causation.

---

# 19. Section 9 — Feature Selection

For the initial model:

```python
features = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Year",
    "Month",
    "Day",
    "WeekOfYear"
]

target = "Sales"
```

The target is:

```text
Sales
```

The model learns a relationship:

```text
Features → Sales
```

This is supervised learning.

Because `Sales` is continuous numerical data, this is a regression problem.

---

# 20. Section 10 — Categorical Encoding

`StoreType` and `Assortment` are categorical.

Use one-hot encoding:

```python
df_model = pd.get_dummies(
    df[features + [target]],
    columns=["StoreType", "Assortment"],
    drop_first=True
)
```

Inspect:

```python
df_model.head()
```

---

# 21. Missing Values

Check:

```python
df_model.isnull().sum()
```

For a first version:

```python
df_model = df_model.dropna()
```

For a more advanced version, consider explicit imputation and document the method.

---

# 22. Section 11 — Chronological Train/Test Split

This is an important part of the project.

Because the project is forecasting-oriented, avoid randomly mixing past and future observations.

Sort the data:

```python
df_model = df_model.sort_values(
    by=["Year", "Month", "Day"]
)
```

Separate features and target:

```python
X = df_model.drop(columns=["Sales"])
y = df_model["Sales"]
```

Create an 80/20 chronological split:

```python
split_index = int(len(df_model) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
```

Conceptually:

```text
Earlier observations
████████████████████████████████
            TRAIN

Later observations
████████████
           TEST
```

### Why not random splitting?

A random split can allow observations from later periods to enter the training data while earlier periods are in the test data.

For forecasting-style evaluation, keeping the test set later in time better represents predicting future observations.

---

# 23. Section 12 — Linear Regression Baseline

Linear Regression is the baseline.

```python
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)
```

Predict:

```python
y_pred_linear = linear_model.predict(X_test)
```

---

# 24. Linear Regression Evaluation

## MAE

```python
mae_linear = mean_absolute_error(
    y_test,
    y_pred_linear
)

print("Linear Regression MAE:", mae_linear)
```

## RMSE

```python
rmse_linear = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_linear
    )
)

print("Linear Regression RMSE:", rmse_linear)
```

## R²

```python
r2_linear = r2_score(
    y_test,
    y_pred_linear
)

print("Linear Regression R²:", r2_linear)
```

---

# 25. Understanding the Metrics

## MAE — Mean Absolute Error

MAE is the average absolute difference between actual and predicted values.

Example:

```text
MAE = 1,000
```

means the model's predictions differ from actual sales by about 1,000 sales units on average.

MAE is easy to explain.

---

## RMSE — Root Mean Squared Error

RMSE also measures prediction error but gives greater weight to larger errors.

Therefore:

- MAE = easy-to-interpret average error
- RMSE = more sensitive to large errors

---

## R² — R-squared

R² measures how much variation in the target is explained by the model relative to a mean-based baseline.

Do not judge a model using R² alone.

---

# 26. Section 13 — Random Forest Regressor

Create the model:

```python
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Train:

```python
rf_model.fit(X_train, y_train)
```

Predict:

```python
y_pred_rf = rf_model.predict(X_test)
```

---

# 27. Random Forest Evaluation

```python
mae_rf = mean_absolute_error(
    y_test,
    y_pred_rf
)

rmse_rf = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred_rf
    )
)

r2_rf = r2_score(
    y_test,
    y_pred_rf
)

print("Random Forest MAE:", mae_rf)
print("Random Forest RMSE:", rmse_rf)
print("Random Forest R²:", r2_rf)
```

---

# 28. Model Comparison

Create:

```python
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        mae_linear,
        mae_rf
    ],
    "RMSE": [
        rmse_linear,
        rmse_rf
    ],
    "R2": [
        r2_linear,
        r2_rf
    ]
})

results
```

Compare:

- MAE
- RMSE
- R²

Do not choose a model solely because it has a higher R².

For this project, discuss the metrics together and explain the trade-offs.

---

# 29. Actual vs Predicted

```python
plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred_rf,
    alpha=0.3
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.show()
```

A useful prediction model should show a meaningful relationship between actual and predicted values.

Also inspect whether errors become larger for higher-sales observations.

---

# 30. Residual Analysis

Calculate residuals:

```python
residuals = y_test - y_pred_rf
```

Plot:

```python
plt.figure(figsize=(10, 5))

plt.scatter(
    y_pred_rf,
    residuals,
    alpha=0.3
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Sales")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.show()
```

### What to look for

Ideally, residuals should not show obvious systematic patterns.

If there is a strong pattern, the model may be missing some relationship in the data.

---

# 31. Feature Importance

Random Forest provides feature importance.

```python
importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
)
```

Sort:

```python
importance = importance.sort_values(
    ascending=False
)
```

Inspect:

```python
importance.head(15)
```

Plot:

```python
plt.figure(figsize=(10, 6))

importance.head(15).sort_values().plot(
    kind="barh"
)

plt.title("Top Features Contributing to Model Predictions")
plt.xlabel("Feature Importance")

plt.show()
```

### Important wording

Do not write:

> These features cause sales.

Write:

> These features contributed most strongly to the Random Forest's predictions according to the model's feature-importance measure.

Feature importance is not causal evidence.

---

# 32. Section 14 — SQL Analysis

Create:

```text
sql/
└── analysis.sql
```

Example queries:

```sql
-- Average sales by store

SELECT
    Store,
    AVG(Sales) AS avg_sales
FROM sales
GROUP BY Store
ORDER BY avg_sales DESC;
```

Promotion:

```sql
-- Average sales by promotion status

SELECT
    Promo,
    AVG(Sales) AS avg_sales
FROM sales
GROUP BY Promo;
```

Store type:

```sql
-- Average sales by store type

SELECT
    StoreType,
    AVG(Sales) AS avg_sales
FROM sales
GROUP BY StoreType
ORDER BY avg_sales DESC;
```

Monthly analysis:

```sql
-- Average sales by month

SELECT
    Month,
    AVG(Sales) AS avg_sales
FROM sales
GROUP BY Month
ORDER BY Month;
```

### Important

The exact SQL environment is up to you.

If you use SQLite, PostgreSQL, MySQL, DuckDB, etc., document which one you used.

The goal is to demonstrate practical SQL analysis, not to build a complicated database.

---

# 33. Optional Section — Store Segmentation with K-Means

Do this only after the core regression project works.

This adds clustering to the project.

Business question:

> Can stores be grouped according to their sales and operating characteristics?

Create store-level features:

```python
store_features = df.groupby("Store").agg({
    "Sales": "mean",
    "Customers": "mean",
    "Promo": "mean",
    "CompetitionDistance": "mean"
}).reset_index()
```

Import:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

Prepare:

```python
X_cluster = store_features.drop(
    columns=["Store"]
)
```

Scale:

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_cluster
)
```

Create clusters:

```python
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

store_features["Cluster"] = kmeans.fit_predict(
    X_scaled
)
```

Inspect:

```python
store_features.groupby("Cluster").mean()
```

The clusters should be described based on their measured characteristics.

Do not automatically label them "good" or "bad".

---

# 34. Business Insights

This is one of the most important sections.

Do not write generic observations.

Use your actual results.

Answer:

### 1. What sales patterns did you observe?

Example areas:

- Time
- Month
- Weekday
- Store type

### 2. What did you observe about promotions?

Compare observed average sales between promotional and non-promotional periods.

### 3. Which features mattered most to the model?

Use feature importance.

### 4. Which model performed better under the selected evaluation metrics?

Report the actual numbers.

### 5. How could predictions potentially help a retailer?

Possible applications:

- Inventory planning
- Staff planning
- Promotion planning
- Operational planning
- Demand estimation

Do not claim that the model automatically solves inventory optimization. Sales forecasting is only one input into such decisions.

---

# 35. Example Business Insights Template

After completing the analysis, replace this template with your actual results:

```markdown
## Business Insights

1. Sales showed observable variation across time periods, suggesting
   temporal patterns relevant to demand planning.

2. Promotional and non-promotional periods showed different average
   observed sales levels.

3. Sales behavior varied across store types, indicating that store-level
   characteristics may be useful predictive variables.

4. The Random Forest model identified several features that contributed
   strongly to its predictions.

5. The forecasting approach could potentially support inventory and
   operational planning by providing estimated future sales.
```

Only keep statements that are supported by your actual analysis.

---

# 36. Model Limitations

Include:

```markdown
## Limitations

- The dataset represents historical sales and may not fully represent
  future market conditions.
- External variables such as weather, local economic conditions,
  competitor actions, and unexpected events are not fully captured.
- The model predicts sales but does not establish causal relationships.
- Additional lag and rolling-window features could potentially improve
  the forecasting approach.
- Model performance may vary across individual stores and time periods.
```

---

# 37. Future Improvements

```markdown
## Future Improvements

- Add lag-based sales features
- Add rolling averages
- Experiment with gradient boosting models
- Perform hyperparameter tuning
- Incorporate additional external variables
- Evaluate models using rolling/expanding time-series validation
- Develop a formal time-series forecasting approach
- Deploy the model through an API
- Build an interactive dashboard
```

Do not claim that these improvements were implemented unless you actually implement them.

---

# 38. Final Notebook Structure

Your finished notebook should look like:

```text
# Retail Sales & Demand Forecasting

## 1. Business Problem
## 2. Objective
## 3. Dataset
## 4. Import Libraries
## 5. Load Data
## 6. Data Understanding
## 7. Data Cleaning
## 8. Merge Data
## 9. Exploratory Data Analysis
### 9.1 Sales Distribution
### 9.2 Sales Over Time
### 9.3 Sales by Day
### 9.4 Promotion Analysis
### 9.5 Store Type Analysis
### 9.6 Monthly Analysis
### 9.7 Correlation Analysis

## 10. Feature Engineering
## 11. Feature Selection
## 12. Encoding
## 13. Train/Test Split

## 14. Linear Regression
### 14.1 Training
### 14.2 Predictions
### 14.3 Evaluation

## 15. Random Forest
### 15.1 Training
### 15.2 Predictions
### 15.3 Evaluation

## 16. Model Comparison
## 17. Actual vs Predicted
## 18. Residual Analysis
## 19. Feature Importance

## 20. SQL Analysis

## 21. Business Insights
## 22. Limitations
## 23. Future Improvements

## 24. Optional Store Segmentation
```

---

# 39. GitHub README Structure

Your `README.md` should contain:

```markdown
# Retail Sales & Demand Forecasting

## Overview

## Business Problem

## Objective

## Dataset

## Technologies Used

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- SQL
- Git/GitHub

## Project Workflow

## Exploratory Data Analysis

## Feature Engineering

## SQL Analysis

## Machine Learning

### Linear Regression

### Random Forest

## Model Evaluation

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | ... | ... | ... |
| Random Forest | ... | ... | ... |

## Business Insights

## Limitations

## Future Improvements

## How to Run

## Project Structure
```

---

# 40. requirements.txt

Use:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Add any additional library only if you actually use it.

---

# 41. GitHub Repository Structure

Final structure:

```text
retail-demand-forecasting/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── retail_demand_forecasting.ipynb
│
├── sql/
│   └── analysis.sql
│
├── images/
│   ├── sales_distribution.png
│   ├── sales_over_time.png
│   ├── sales_by_day.png
│   ├── promotion_analysis.png
│   └── actual_vs_predicted.png
│
├── README.md
├── PROJECT_GUIDE.md
└── requirements.txt
```

---

# 42. Two-Day Execution Plan

## DAY 1 — Data + EDA

### Morning

- Download dataset
- Create folders
- Create notebook
- Load datasets
- Inspect data
- Check missing values
- Check duplicates
- Merge datasets
- Handle closed stores

### Afternoon

- Convert dates
- Create date features
- Sales distribution
- Sales over time
- Sales by weekday
- Promotion analysis
- Store type analysis
- Monthly analysis
- Correlation matrix

### Evening

- Feature selection
- Categorical encoding
- Missing-value handling
- Chronological train/test split
- Start SQL analysis

---

# 43. DAY 2 — Machine Learning + Portfolio

## Morning

- Linear Regression
- MAE
- RMSE
- R²
- Random Forest
- MAE
- RMSE
- R²
- Model comparison

## Afternoon

- Actual vs predicted plot
- Residual analysis
- Feature importance
- Business insights
- Limitations
- Future improvements

If time remains:

- K-Means store segmentation

## Evening

- Clean notebook
- Add explanations
- Save charts
- Write README
- Add requirements.txt
- Create GitHub repository
- Push project
- Prepare resume bullet points
- Prepare interview questions

---

# 44. Interview Questions You Should Be Able to Answer

Before putting this project on your resume, make sure you can explain:

### Basic

1. What is the business problem?
2. What is the target variable?
3. Why is this a regression problem?
4. What is supervised learning?
5. What features did you use?

### Data

6. How did you handle missing values?
7. Why did you merge the two datasets?
8. Why did you remove closed stores?
9. What is feature engineering?
10. Why did you extract month/day/week from the date?

### Statistics

11. What is mean vs median?
12. What is correlation?
13. Does correlation imply causation?
14. What is an outlier?

### Machine Learning

15. Why Linear Regression?
16. Why Random Forest?
17. What is a decision tree?
18. What is an ensemble?
19. What is overfitting?
20. How could you reduce overfitting?

### Evaluation

21. What is MAE?
22. What is RMSE?
23. Why does RMSE penalize large errors more?
24. What is R²?
25. Which metric would you focus on for this business problem and why?

### Time Series

26. Why did you use a chronological train/test split?
27. What is data leakage?
28. What are lag features?
29. What are rolling averages?
30. How would you improve the forecasting approach?

### Business

31. What did you actually discover?
32. How could a retailer use these predictions?
33. What are the limitations?
34. What additional data would you want?
35. What would you do with another month?

### SQL

36. How did you use SQL?
37. What is GROUP BY?
38. What is an aggregate function?
39. Difference between WHERE and HAVING?
40. What is a JOIN?

---

# 45. Resume Entry

Only after the project is actually completed, use your real results.

Template:

```text
Retail Sales & Demand Forecasting | Python, SQL, Scikit-learn
• Analyzed retail sales data using Python and SQL to identify temporal,
  promotional, and store-level sales patterns.
• Engineered time-based and store-level features and developed regression
  models using Linear Regression and Random Forest.
• Evaluated model performance using MAE, RMSE, and R² and analyzed model
  feature importance to derive business insights for demand planning.
```

Do not invent model performance numbers.

If the final project includes clustering, add it only if you genuinely complete and understand it.

---

# 46. What NOT to Do

Avoid:

- Randomly generating 20+ charts
- Copying Kaggle notebooks without understanding them
- Claiming causation from correlation
- Claiming feature importance means causation
- Randomly splitting forecasting data without explaining it
- Reporting only R²
- Inventing performance numbers
- Adding deep learning just to make the project look advanced
- Adding clustering without a business reason
- Claiming the model performs "perfectly"
- Uploading huge raw datasets unnecessarily
- Writing a README full of buzzwords you cannot explain

The goal is not to make the project look complicated.

The goal is to make it **understandable, technically sound, and defensible in an interview**.

---

# 47. Final Definition of Done

The project is finished when you have:

- [ ] Clean notebook
- [ ] Dataset loaded and understood
- [ ] Data merged
- [ ] Missing values handled
- [ ] Closed stores handled
- [ ] Date features created
- [ ] At least 5 meaningful EDA visualizations
- [ ] Correlation analysis
- [ ] SQL analysis
- [ ] Chronological train/test split
- [ ] Linear Regression baseline
- [ ] Random Forest model
- [ ] MAE calculated
- [ ] RMSE calculated
- [ ] R² calculated
- [ ] Model comparison table
- [ ] Actual vs predicted plot
- [ ] Residual analysis
- [ ] Feature importance
- [ ] Business insights based on actual results
- [ ] Limitations
- [ ] Future improvements
- [ ] README
- [ ] requirements.txt
- [ ] GitHub repository
- [ ] Resume bullet points
- [ ] Interview preparation

---

# 48. Recommended Order for Building It

Do not jump around.

Follow this exact order:

```text
1. Download Rossmann dataset
2. Create project folders
3. Create notebook
4. Load train.csv
5. Load store.csv
6. Inspect both datasets
7. Merge them
8. Handle closed stores
9. Convert Date
10. Create date features
11. Perform EDA
12. Select features
13. Encode categorical variables
14. Handle missing values
15. Sort chronologically
16. Create train/test split
17. Train Linear Regression
18. Evaluate Linear Regression
19. Train Random Forest
20. Evaluate Random Forest
21. Compare models
22. Plot actual vs predicted
23. Analyze residuals
24. Analyze feature importance
25. Perform SQL analysis
26. Write business insights
27. Write limitations
28. Add optional clustering if time permits
29. Clean notebook
30. Write README
31. Push to GitHub
32. Add project to resume
33. Prepare for interview
```

**Do the core regression project first.** The SQL and optional clustering sections should strengthen the project, not delay getting the main pipeline working.
