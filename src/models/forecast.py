from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def simple_time_forecast(series: pd.Series, horizon: int = 4) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).reset_index(drop=True)
    if len(values) < 2:
        future = pd.DataFrame({"step": list(range(1, horizon + 1)), "forecast": [float(values.iloc[-1]) if len(values) else 0.0] * horizon})
        return future

    X = np.arange(len(values)).reshape(-1, 1)
    y = values.to_numpy()
    model = LinearRegression()
    model.fit(X, y)

    future_steps = np.arange(len(values), len(values) + horizon).reshape(-1, 1)
    preds = model.predict(future_steps)
    return pd.DataFrame({"step": list(range(1, horizon + 1)), "forecast": preds})
