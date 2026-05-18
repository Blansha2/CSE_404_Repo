# Weather Forecasting with Machine Learning

## Project Overview

This project explores machine learning and deep learning approaches for short-term temperature forecasting using historical weather data collected across multiple cities in North Carolina.

The primary objective was to predict hourly temperature patterns using meteorological measurements, engineered temporal features, and lag-based forecasting techniques.

Models explored include:
- Linear Regression
- Ridge Regression
- Lasso Regression
- XGBoost
- LSTM Neural Network

The project compares model performance using forecasting metrics such as:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Motivation

Weather forecasting is an important real-world time series problem with applications in:
- agriculture
- transportation
- energy demand forecasting
- logistics planning
- disaster preparedness

This project investigates how different machine learning approaches perform on sequential weather data and how feature engineering techniques impact forecasting accuracy.

---

## Team Context

This project originated as a college group project for a machine learning/data science course. While the project topic and planning were discussed collaboratively, I independently designed and implemented the full technical pipeline, including:

- data preprocessing
- exploratory data analysis
- feature engineering
- visualization
- model development
- hyperparameter tuning
- forecasting evaluation
- model comparison

---

## Dataset

Dataset Source:  
https://huggingface.co/datasets/Katherinetian/weather_data_NC

The dataset contains historical weather observations across multiple North Carolina cities, including:
- temperature
- humidity
- pressure
- wind speed
- cloud coverage
- weather conditions

Cities included:
- Asheville
- Raleigh
- Charlotte
- Durham
- Wilmington
- Greensboro
- Greenville
- New Bern
- Chapel Hill / Winnabow

---

## Data Cleaning

Data preprocessing steps included:
- removing highly sparse columns
- converting date columns into datetime format
- handling categorical weather conditions
- one-hot encoding categorical variables
- sorting chronologically for time-series forecasting

Columns with excessive missing values were removed, including:
- wind gust
- rain measurements
- snow measurements

---

## Exploratory Data Analysis

Initial exploratory analysis included:
- temperature trend visualization
- correlation heatmaps
- weather condition frequency analysis

This analysis helped identify:
- strong relationships between temperature-related variables
- highly correlated features
- potential redundant predictors

---

## Feature Engineering

Several time-series feature engineering techniques were implemented to improve forecasting performance.

### Temporal Features
- year
- quarter
- month
- week
- day

### Cyclical Encoding
Sine and cosine transformations were applied to cyclical variables to preserve periodic relationships in time data.

Examples:
- month_sin
- month_cos
- day_sin
- day_cos

### Lag Features
Historical temperature observations were used as predictors:
- lag_1
- lag_2
- lag_7

These lag features became some of the strongest predictors in the final forecasting model.

---

## Models Used

### Linear Models
- Linear Regression
- Ridge Regression
- Lasso Regression

### Ensemble Learning
- XGBoost Regressor

### Deep Learning
- LSTM Neural Network using TensorFlow/Keras

---

## Model Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 2.45 | 3.24 | 0.767 |
| Lasso Regression | 2.25 | 3.22 | 0.770 |
| Ridge Regression | 2.45 | 3.24 | 0.767 |
| XGBoost | 2.33 | 3.18 | 0.804 |
| LSTM | 2.82 | 3.76 | 0.685 |

The optimized XGBoost model achieved the strongest overall forecasting performance.

---

## Cross-City Generalization Testing

To evaluate model robustness and generalization, the final XGBoost model trained on Asheville weather data was tested on unseen weather data from additional North Carolina cities.

The model maintained strong forecasting performance across multiple cities, suggesting that the learned temporal and meteorological patterns generalized effectively beyond the original training location.

---

## XGBoost Feature Importance

Feature importance analysis revealed that lag-based forecasting features were the strongest predictors of future temperature.

### Most Important Features
1. `lag_1`
2. `lag_2`
3. `month_cos`
4. `lag_7`

This indicates that:
- recent temperature history was highly predictive
- seasonal cyclical patterns strongly influenced forecasting performance
- short-term temporal dependencies played a major role in temperature prediction

Meteorological variables such as:
- humidity
- wind speed
- cloud coverage
- pressure

also contributed to model performance, though with lower relative importance.

---

## Key Takeaways

- Feature engineering significantly improved forecasting accuracy
- Lag-based features were highly predictive
- Ensemble tree methods outperformed both linear models and LSTM architectures
- XGBoost generalized effectively across multiple cities
- Deep learning models did not outperform gradient boosting methods on this dataset

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- TensorFlow / Keras
- matplotlib
- seaborn

---

## Future Improvements

Potential future enhancements include:
- walk-forward time-series validation
- SHAP explainability analysis
- rolling statistical forecasting features
- deployment with Streamlit
- additional forecasting models such as SARIMA or Prophet

---

## Repository Structure

```text
weather-forecasting-ml/
│
├── data/
├── notebooks/
├── plots/
├── README.md
└── requirements.txt
```

---

## Author

Ashlynn Blanshan  
Data Science Student — Michigan State University
