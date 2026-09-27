# Retail Sales & Demand Forecasting

## Overview

This is an end-to-end data science case study analyzing historical retail store sales to predict future daily sales. It demonstrates exploratory data analysis, data cleaning, feature engineering, SQL analysis, and predictive modeling using regression techniques.

## Business Problem

Retailers need to estimate future sales to support inventory planning, promotional decisions, and overall operational planning.

## Objective

Analyze historical retail sales data and develop machine learning models to predict daily store sales. We utilize Store characteristics, Promotions, Holidays, Day of week, Month/year, and Competition-related information.

## Dataset

Use the Rossmann Store Sales dataset. We'll need `train.csv` and `store.csv` placed in the `data/` directory.

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

1. Data Understanding & Cleaning
2. Exploratory Data Analysis
3. Feature Engineering & Selection
4. SQL Analysis
5. Chronological Train/Test Split
6. Machine Learning (Linear Regression, Random Forest)
7. Model Evaluation
8. Business Interpretation

## Exploratory Data Analysis

Explored sales distribution, patterns across time, day of week, promotional periods, and store types. Uncovered meaningful temporal and categorical trends to guide modeling.

## Feature Engineering

Extracted temporal features (Year, Month, Day, DayOfWeek, WeekOfYear) from Date, applied one-hot encoding to categorical features like StoreType and Assortment, and filtered out non-operational days.

## SQL Analysis

Used SQL aggregations to evaluate sales by store, promotion status, store type, and month. (See `sql/analysis.sql`).

## Machine Learning

### Linear Regression

Baseline model trained on 80% chronologically split data to establish preliminary metrics.

### Random Forest

Ensemble model configured with 100 estimators to capture non-linear relationships.

## Model Evaluation

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1985.72 | 2737.52 | 0.20 |
| Random Forest | 738.01 | 1098.73 | 0.87 |


## Business Insights

1. Sales showed observable variation across time periods, suggesting temporal patterns relevant to demand planning.
2. Promotional and non-promotional periods showed different average observed sales levels.
3. Sales behavior varied across store types, indicating that store-level characteristics may be useful predictive variables.
4. The Random Forest model identified several features that contributed strongly to its predictions.
5. The forecasting approach could potentially support inventory and operational planning by providing estimated future sales.

## Limitations

- The dataset represents historical sales and may not fully represent future market conditions.
- External variables such as weather, local economic conditions, competitor actions, and unexpected events are not fully captured.
- The model predicts sales but does not establish causal relationships.
- Additional lag and rolling-window features could potentially improve the forecasting approach.
- Model performance may vary across individual stores and time periods.

## Project Structure

```text
retail-demand-forecasting/
│
├── data/
│   ├── train.csv
│   └── test.csv
│   └── store.csv
│
├── images/
│   └── actual_vs_predicted.png
│   └── promotion_analysis.png
│   └── sales_by_day.png
│   └── sales_distribuiton.png
│   └── sales_over_time.png
│
├── notebooks/
│   └── retail_demand_forecasting.ipynb
│   └── retail_demand_forecasting.md
│   └── retail_demand_forecasting_files/
│
├── sql/
│   └── analysis.sql
│
└── README.md
```
