import shap
import pandas as pd
import joblib


class CreditDefaultModel:
    def __init__(self, model_path="models/best_model.pkl"):
        self.model = joblib.load(model_path)
        self.explainer = shap.Explainer(self.model)

    def predict(self, data: dict):
        df = pd.DataFrame([data])
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]

        return {
            "prediction": int(prediction),
            "default_probability": float(probability)
        }

    def explain(self, data: dict):
        df = pd.DataFrame([data])
        shap_values = self.explainer(df)

        explanation = dict(zip(df.columns, shap_values.values[0]))

        return explanation