# 💳 Credit Default Prediction System

Production-grade machine learning system for predicting credit card default risk using classical ML + MLOps best practices.

This project includes:

* Model training with hyperparameter tuning
* MLflow experiment tracking & model registry
* Model versioning
* FastAPI inference service
* SHAP explainability endpoint
* Logging & monitoring support
* Dockerized deployment
* CI/CD with GitHub Actions

---

# 📌 Problem Statement

Predict whether a customer will default on their credit card payment next month using financial history, demographic attributes, and repayment behavior.

Dataset: Default of Credit Card Clients (UCI ML Repository)

Target:

* `1` → Default
* `0` → No default

---

# 🏗 System Architecture

```
credit-default-project/
│
├── app/
│   ├── main.py              # FastAPI app
│   ├── schema.py            # Request/response validation
│   ├── logger.py            # Structured logging
│
├── src/
│   ├── train.py             # Training + MLflow tracking
│   ├── predict.py           # Inference + SHAP
│
├── models/
│   └── best_model.pkl
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 🚀 Features

## ✅ Machine Learning

* Stratified Train/Test split
* Cross-validation
* Hyperparameter tuning
* ROC-AUC optimization
* Class imbalance handling

## ✅ MLOps

* MLflow experiment tracking
* Model registry & versioning
* Structured logging
* SHAP explainability
* CI/CD pipeline

## ✅ Deployment

* FastAPI REST API
* Docker container
* Swagger auto documentation

---

# 🧠 MLflow Experiment Tracking

Start MLflow UI:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

Tracked:

* Parameters
* Metrics (ROC-AUC)
* Model artifacts
* Model versions

Registered Model Name:

```
CreditDefaultModel
```

---

# 🏋️ Training the Model

Run:

```bash
python src/train.py
```

This will:

* Perform hyperparameter tuning
* Log experiments to MLflow
* Register model version
* Save best model to `/models`

---

# 🌐 Running the API Locally

Start the API:

```bash
uvicorn app.main:app --reload
```

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

# 🔮 API Endpoints

## 1️⃣ Predict Default

POST `/predict`

Request body:

```json
{
  "X1": 20000,
  "X2": 2,
  "X3": 2,
  "X4": 1,
  "X5": 24,
  "X6": 2,
  "X7": 2,
  "X8": -1,
  "X9": -1,
  "X10": -2,
  "X11": -2,
  "X12": 3913,
  "X13": 3102,
  "X14": 689,
  "X15": 0,
  "X16": 0,
  "X17": 0,
  "X18": 0,
  "X19": 689,
  "X20": 0,
  "X21": 0,
  "X22": 0,
  "X23": 0
}
```

Response:

```json
{
  "prediction": 1,
  "default_probability": 0.82
}
```

---

## 2️⃣ Explain Prediction (SHAP)

POST `/explain`

Returns feature-level contribution scores.

Example response:

```json
{
  "X6": 0.45,
  "X1": -0.12,
  "X12": 0.31,
  ...
}
```

---

# 🐳 Docker Deployment

Build:

```bash
docker build -t credit-default-api .
```

Run:

```bash
docker run -p 8000:8000 credit-default-api
```

---

# 🔁 CI/CD Pipeline

GitHub Actions automatically:

* Installs dependencies
* Runs training script
* Fails build on errors

Workflow file:

```
.github/workflows/ci.yml
```

---

# 📊 Monitoring & Logging

* Structured logs via Python logging
* Ready to integrate with:

  * ELK Stack
  * Datadog
  * Prometheus + Grafana

Every API request and prediction is logged.

---

# 🔍 Explainability

This system includes SHAP explainability.

Why it matters:

* Required for financial regulation
* Supports model transparency
* Enables risk team audits
* Detects bias patterns

---

# 📈 Future Enhancements

* Model drift detection
* Automated retraining pipeline
* Batch prediction endpoint
* Feature store integration
* Cloud deployment (AWS ECS / GCP / Azure)
* Authentication layer (JWT)
* Rate limiting

---

# 🛠 Requirements

Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📜 License

For educational and demonstration purposes.

---

# 👨‍💻 Author

Built as a production-grade ML system demonstrating:

* ML engineering
* Model lifecycle management
* API deployment
* MLOps practices

---

If you’d like, I can also generate:

* 🔥 A resume-ready project description
* 🔥 System architecture diagram
* 🔥 AWS deployment guide
* 🔥 Model drift detection module
* 🔥 Terraform infrastructure setup

What direction do you want to scale this next?
