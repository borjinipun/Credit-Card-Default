import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


mlflow.set_experiment("credit_default_experiment")


def load_data():
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features
    y = dataset.data.targets.values.ravel()
    return X, y


def build_models():
    return {
        "LogisticRegression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000))
            ]),
            {
                "model__C": [0.1, 1],
                "model__class_weight": [None, "balanced"]
            }
        ),
        "RandomForest": (
            Pipeline([
                ("model", RandomForestClassifier(random_state=42))
            ]),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10]
            }
        ),
        "XGBoost": (
            Pipeline([
                ("model", XGBClassifier(
                    eval_metric="logloss",
                    random_state=42
                ))
            ]),
            {
                "model__n_estimators": [100],
                "model__learning_rate": [0.1],
                "model__max_depth": [3, 5]
            }
        )
    }


def train():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = build_models()
    best_score = 0
    best_model = None
    best_name = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, (pipeline, param_grid) in models.items():

        with mlflow.start_run(run_name=name):

            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
            test_roc = roc_auc_score(y_test, y_proba)

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("test_roc_auc", test_roc)

            mlflow.sklearn.log_model(
                grid.best_estimator_,
                "model",
                registered_model_name="CreditDefaultModel"
            )

            if test_roc > best_score:
                best_score = test_roc
                best_model = grid.best_estimator_
                best_name = name

    joblib.dump(best_model, "models/best_model.pkl")

    print("Best model:", best_name)
    print("ROC-AUC:", best_score)


if __name__ == "__main__":
    train()