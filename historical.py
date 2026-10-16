import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Predictive Analytics", page_icon="📈", layout="wide")
st.title("📈 Predictive Analytics Using Historical Data")
st.write("Explore historical sales, train a regression model, and forecast future trends.")

uploaded = st.sidebar.file_uploader("Upload CSV (date and sales columns)", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)
else:
    raw = pd.read_csv("data/historical_sales.csv")

required = {"date", "sales"}
if not required.issubset(raw.columns):
    st.error("CSV must contain columns named 'date' and 'sales'.")
    st.stop()

raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw["sales"] = pd.to_numeric(raw["sales"], errors="coerce")
data = raw.dropna(subset=["date", "sales"]).sort_values("date")
data = data.groupby("date", as_index=False)["sales"].sum()

if len(data) < 8:
    st.error("Please provide at least 8 valid dated observations.")
    st.stop()

data["time_index"] = np.arange(len(data))
X = data[["time_index"]]
y = data["sales"]

# Chronological holdout: last 20% for evaluation, earlier observations for training.
split = max(5, int(len(data) * 0.8))
if split >= len(data):
    split = len(data) - 1
model = LinearRegression()
model.fit(X.iloc[:split], y.iloc[:split])
test_predictions = model.predict(X.iloc[split:])
mae = mean_absolute_error(y.iloc[split:], test_predictions)
rmse = np.sqrt(mean_squared_error(y.iloc[split:], test_predictions))
r2 = r2_score(y.iloc[split:], test_predictions) if len(test_predictions) > 1 else float("nan")

# Fit final model on all available history for forecasting.
final_model = LinearRegression().fit(X, y)
future_count = st.sidebar.slider("Forecast periods", min_value=1, max_value=24, value=6)
future_index = np.arange(len(data), len(data) + future_count)
future_sales = final_model.predict(future_index.reshape(-1, 1))
freq = pd.infer_freq(data["date"])
if freq:
    future_dates = pd.date_range(data["date"].iloc[-1], periods=future_count + 1, freq=freq)[1:]
else:
    # For irregular dates, forecast at the median observed interval.
    median_days = max(1, int(data["date"].diff().dt.days.median()))
    future_dates = pd.to_datetime(data["date"].iloc[-1]) + pd.to_timedelta(median_days * np.arange(1, future_count + 1), unit="D")

forecast = pd.DataFrame({"date": future_dates, "predicted_sales": np.maximum(0, future_sales).round(2)})

c1, c2, c3 = st.columns(3)
c1.metric("Historical records", f"{len(data):,}")
c2.metric("Validation MAE", f"{mae:,.2f}")
c3.metric("Validation RMSE", f"{rmse:,.2f}")

st.subheader("Historical sales")
st.line_chart(data.set_index("date")["sales"])

st.subheader("Forecast")
plot_data = pd.DataFrame({
    "Historical": data.set_index("date")["sales"],
    "Forecast": forecast.set_index("date")["predicted_sales"]
})
st.line_chart(plot_data)

st.subheader("Forecast table")
st.dataframe(forecast, use_container_width=True)
st.download_button("Download forecast CSV", forecast.to_csv(index=False), "sales_forecast.csv", "text/csv")

with st.expander("Model evaluation details"):
    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² on chronological holdout: {r2:.3f}")
    st.caption("These are holdout metrics, not a guarantee of future performance. A linear trend model may miss seasonality and sudden market changes.")
