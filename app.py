#####################################################################
#### app.py ####

from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib
import os
from scipy import stats

app = Flask(__name__)
model_path = "regression_model.pkl"

@app.route("/estimate_ate", methods=["POST"])
def estimate_ate():
    data = request.get_json()
    df = pd.DataFrame(data)

    X = df[["Treatment_W", "Sustainability_Spending_X"]]
    y = df["Engagement_Score_Y"]

    model = LinearRegression().fit(X, y)
    joblib.dump(model, model_path)

    # Compute statistical significance (standard error and p-value)
    n, k = X.shape
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - k - 1)
    X_with_intercept = np.column_stack([np.ones(n), X])
    var_b = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept).diagonal()
    se_b = np.sqrt(var_b)
    t_stat_tau = model.coef_[0] / se_b[1]
    p_value_tau = 2 * (1 - stats.t.cdf(np.abs(t_stat_tau), df=n - k - 1))

    # Log model coefficients and p-value
    with open("output.txt", "w") as f:
        f.write(f"Intercept (alpha): {model.intercept_}\n")
        f.write(f"Coefficients: {model.coef_.tolist()}\n")
        f.write(f"ATE (tau): {model.coef_[0]}\n")
        f.write(f"Standard error (tau): {se_b[1]}\n")
        f.write(f"p-value (tau): {p_value_tau}\n")

    result = {
        "alpha": model.intercept_,
        "tau (ATE)": model.coef_[0],
        "beta": model.coef_[1],
        "p_value_tau": p_value_tau
    }
    return jsonify(result)

@app.route("/predict_engagement")
def predict_engagement():
    W = float(request.args.get("W", 0))
    X_val = float(request.args.get("X", 0))

    if not os.path.exists(model_path):
        return jsonify({"error": "Model not trained. Please POST to /estimate_ate first."})

    model = joblib.load(model_path)
    y_pred = model.predict([[W, X_val]])[0]

    with open("output.txt", "a") as f:
        f.write(f"Prediction for W={W}, X={X_val}: {y_pred}\n")

    return jsonify({"W": W, "X": X_val, "predicted_engagement_Y": round(y_pred, 2)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
