# Predictive Analytics Using Historical Data

## Aim
Build a small application that learns from historical sales data and forecasts future sales trends.

## Features
- Upload a CSV containing `date` and `sales` columns.
- Clean invalid dates and non-numeric sales values.
- Aggregate duplicate dates.
- Train a Linear Regression model.
- Evaluate using a chronological holdout split and report MAE, RMSE, and R².
- Forecast 1–24 future periods.
- Visualize historical data and forecasts.
- Download forecast results as CSV.

## Project structure
```text
predictive_analytics_project/
├── app.py
├── requirements.txt
├── README.md
├── report.md
└── data/
    └── historical_sales.csv
```

## Run on your computer
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. (Recommended) Create and activate a virtual environment.
4. Install packages:
   `pip install -r requirements.txt`
5. Start the app:
   `streamlit run app.py`
6. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Dataset format
CSV headers must be `date,sales`, for example:
```csv
date,sales
2024-01-01,2500
2024-02-01,2640
```
At least 8 valid observations are needed. The included dataset is synthetic sample data created for demonstration, not real company data.

## Algorithm
Linear Regression models sales as a linear function of a time index:
`predicted_sales = intercept + coefficient × time_index`

The latest 20% of observations (chronologically) are held out for validation. After metrics are calculated, the model is retrained on all available data before generating the future forecast.

## Limitations
- A simple linear regression captures trend but not complex seasonal patterns.
- Results depend on data quality and history length.
- Forecasts are estimates, not guarantees.
