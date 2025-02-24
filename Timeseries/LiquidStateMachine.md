# **Liquid State Machine for Time Series Forecasting (Ocean Data)** 🌊  

## **Overview**  
This project demonstrates how to use a **Liquid State Machine (LSM)** for **time series forecasting** of **Sea Level Anomaly (SLA)** data. The LSM is a type of **reservoir computing model**, which is particularly effective for **predicting sequential data patterns**.  

## **Why Use LSM?** 🤖  
- **Efficient for Time Series**: Captures temporal dependencies in oceanographic data.  
- **Low Computational Cost**: Requires fewer training samples than deep learning models.  
- **Robust for Noisy Data**: Works well with real-world ocean datasets.  

---

## **Dataset 📊**  
The dataset contains **daily Sea Level Anomaly (SLA)** measurements.  

**Columns:**  
- `Dates`: Date of measurement (YYYY-MM-DD)  
- `SLA`: Sea Level Anomaly (normalized)  

Example data:  
```csv
Dates,SLA
2011-01-01,0.0928
2011-01-02,0.0961
2011-01-03,0.1005
...
```

---

## **Installation & Requirements**  
Install the required Python libraries:  
```bash
pip install numpy pandas matplotlib reservoirpy scikit-learn optuna
```

---

## **Model Training & Forecasting 🚀**  

### **1️⃣ Load Data & Preprocess**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("sla_data.csv", parse_dates=["Dates"])

# Normalize SLA data
df["SLA"] = (df["SLA"] - df["SLA"].mean()) / df["SLA"].std()
```

---

### **2️⃣ Train Liquid State Machine**
```python
from reservoirpy.nodes import Reservoir, Ridge

# Define LSM model
reservoir = Reservoir(100, lr=0.9, sr=1.2)
readout = Ridge(alpha=1e-6)

# Train the LSM
X_train = df["SLA"].values[:-1].reshape(-1, 1)
Y_train = df["SLA"].values[1:].reshape(-1, 1)
reservoir_states = reservoir.run(X_train)
readout.fit(reservoir_states, Y_train)
```

---

### **3️⃣ Forecast Next 7 Days**
```python
# Predict next 7 days
future_predictions = []
last_input = df["SLA"].values[-1].reshape(1, 1)

for _ in range(7):
    reservoir_state = reservoir.run(last_input)
    next_sla = readout.predict(reservoir_state)
    future_predictions.append(next_sla.item())
    last_input = next_sla.reshape(1, 1)

# Display results
future_dates = pd.date_range(start=df["Dates"].iloc[-1] + pd.Timedelta(days=1), periods=7)
future_df = pd.DataFrame({"Dates": future_dates, "Predicted_SLA": future_predictions})
print(future_df)
```

---

## **Evaluation Metrics 📉**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

Y_test = df["SLA"].values[1:].reshape(-1, 1)
predicted_sla_test = readout.predict(reservoir_states)

mae = mean_absolute_error(Y_test, predicted_sla_test)
rmse = np.sqrt(mean_squared_error(Y_test, predicted_sla_test))

print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
```

---

## **Results & Visualization 📈**  
```python
plt.figure(figsize=(10, 5))
plt.plot(df["Dates"], df["SLA"], label="Actual SLA", linestyle="dashed", marker="o", color="blue")
plt.plot(future_df["Dates"], future_df["Predicted_SLA"], label="Predicted SLA (Next 7 Days)", marker="o", color="red")
plt.xlabel("Date")
plt.ylabel("SLA (Normalized)")
plt.title("SLA Prediction for Next 7 Days")
plt.legend()
plt.show()
```

---

## **Hyperparameter Optimization (Optional) 🛠**  
For better accuracy, use **Optuna** to optimize LSM hyperparameters:  
```python
import optuna

def objective(trial):
    N = trial.suggest_int("reservoir_size", 50, 300)
    lr = trial.suggest_float("leaking_rate", 0.1, 1.0)
    sr = trial.suggest_float("spectral_radius", 0.5, 1.5)
    alpha = trial.suggest_loguniform("alpha", 1e-8, 1e-2)

    reservoir = Reservoir(N, lr=lr, sr=sr)
    readout = Ridge(alpha=alpha)

    reservoir_states = reservoir.run(X_train)
    readout.fit(reservoir_states, Y_train)

    predicted = readout.predict(reservoir_states)
    return np.sqrt(mean_squared_error(Y_test, predicted))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print(study.best_params)
```

---

## **Next Steps & Improvements 🚀**  
✅ **Improve LSM by tuning spectral radius & leaking rate**  
✅ **Compare with LSTMs / GRUs for long-term forecasting**  
✅ **Incorporate other ocean data (SST, wind speed, pressure, etc.)**  

---

## **References & Credits 📚**  
- Reservoir Computing: [Paper](https://arxiv.org/abs/1710.07452)  
- SLA Data: [Copernicus Marine Service](https://marine.copernicus.eu/)  
- LSM Implementation: [ReservoirPy](https://github.com/reservoirpy/reservoirpy)  

---

## **Contributing 🤝**  
Feel free to **fork, modify, and submit pull requests**! 🚀  

👨‍💻 Developed by: **[PAVAN KUMAR JONNAKUTI]**  
📧 Contact: **pavankumar.j@incois.gov.in**  

