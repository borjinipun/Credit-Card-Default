from fastapi import FastAPI
from app.schema import CreditInput, CreditOutput
from src.predict import CreditDefaultModel
from app.logger import logger

app = FastAPI(title="Credit Default API")

model = CreditDefaultModel()


@app.post("/predict", response_model=CreditOutput)
def predict(data: CreditInput):
    logger.info(f"Incoming request: {data.dict()}")

    result = model.predict(data.dict())

    logger.info(f"Prediction: {result}")

    return result

@app.post("/explain")
def explain(data: CreditInput):
    return model.explain(data.dict())