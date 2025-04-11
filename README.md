# 📊 Causal ATE Estimation API (Repository: causal-ate-api-xm)

This project implements a Flask-based API that estimates and predicts the **Average Treatment Effect (ATE)** using a linear regression model. It accepts input data, trains a model, and provides a prediction endpoint for stakeholder engagement based on sustainability spending and treatment participation.

---

## 🧠 Model

```
Yᵉ = α + τ·Wᵉ + β·Xᵉ + εᵉ
```

Where:
- **Yᵉ**: Engagement Score (observed outcome)
- **Wᵉ**: Treatment indicator (1 = participated, 0 = did not)
- **Xᵉ**: Sustainability Spending ($1,000s)
- **α, τ, β**: Parameters to be estimated
- **τ**: Interpreted as the **Average Treatment Effect (ATE)**

---

## 🚀 Quickstart with Docker

```bash
docker build -t causal-ate-api-xm .
docker run -p 5000:5000 -v $(pwd):/app causal-ate-api-xm
```

---

## 📬 API Endpoints

### `/estimate_ate` [POST]
Train the model using JSON input data:

```bash
curl -X POST http://localhost:5000/estimate_ate \
     -H "Content-Type: application/json" \
     -d @data.json
```

### `/predict_engagement` [GET]
Use the trained model to predict engagement:

```bash
curl "http://localhost:5000/predict_engagement?W=1&X=20"
```

Expected output:
```json
{
  "W": 1.0,
  "X": 20.0,
  "predicted_engagement_Y": 117.07
}
```

---

## 🦜 Component Explanations

This project is composed of modular components that together deliver a robust and reproducible machine learning API:

- **`app.py`** implements the Flask web API. It provides two main routes: `/estimate_ate` trains a linear regression model using input JSON data, and `/predict_engagement` loads the trained model to predict engagement scores based on user inputs.
- **`requirements.txt`** defines the Python package dependencies required to run the app, ensuring consistent environments across development and deployment.
- **`Dockerfile`** containerizes the application, bundling the source code, dependencies, and runtime into a single image that can be reliably run anywhere.
- **Containerization** improves reproducibility by eliminating differences in environments — the app runs identically across all systems that support Docker.

- **`app.py`**: Hosts the Flask API with `/estimate_ate` for model training and `/predict_engagement` for real-time predictions.
- **`requirements.txt`**: Lists required Python libraries including `scikit-learn`, `pandas`, `joblib`, `scipy`, and `flask`.
- **`Dockerfile`**: Builds a consistent container environment for training and inference.
- **Containerization**: Ensures reproducibility and compatibility across systems without local setup.

---

## 📆 Sample Input: `data.json`

```json
[
  {"Unit": 1, "Engagement_Score_Y": 137, "Treatment_W": 0, "Sustainability_Spending_X": 19.8},
  {"Unit": 2, "Engagement_Score_Y": 118, "Treatment_W": 1, "Sustainability_Spending_X": 23.4},
  ...
  {"Unit": 20, "Engagement_Score_Y": 128, "Treatment_W": 1, "Sustainability_Spending_X": 22.8}
]
```
