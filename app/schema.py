from pydantic import BaseModel


class CreditInput(BaseModel):
    X1: int
    X2: int
    X3: int
    X4: int
    X5: int
    X6: int
    X7: int
    X8: int
    X9: int
    X10: int
    X11: int
    X12: int
    X13: int
    X14: int
    X15: int
    X16: int
    X17: int
    X18: int
    X19: int
    X20: int
    X21: int
    X22: int
    X23: int


class CreditOutput(BaseModel):
    prediction: int
    default_probability: float