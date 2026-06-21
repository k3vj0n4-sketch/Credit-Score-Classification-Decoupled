from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import CreditScoreModel

MODEL_PATH = "model_pipeline.pkl"
model_wrapper: Optional[CreditScoreModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_wrapper
    model_wrapper = CreditScoreModel(MODEL_PATH)
    print(f"[startup] Model '{model_wrapper.model_name}' berhasil dimuat dari {MODEL_PATH}")
    yield
    print("[shutdown] API dimatikan.")


app = FastAPI(
    title="Credit Score Classification API",
    description="Backend inferencing model Credit Score (decoupled deployment, dipanggil dari Streamlit)",
    version="1.0.0",
    lifespan=lifespan,
)

class CreditScoreInput(BaseModel):
    Age: float = Field(..., ge=18, le=100, examples=[35])
    Annual_Income: float = Field(..., ge=0, examples=[45000.0])
    Monthly_Inhand_Salary: float = Field(..., ge=0, examples=[3500.0])
    Num_Bank_Accounts: float = Field(..., ge=0, examples=[4])
    Num_Credit_Card: float = Field(..., ge=0, examples=[3])
    Interest_Rate: float = Field(..., ge=0, examples=[12])
    Num_of_Loan: float = Field(..., ge=0, examples=[2])
    Delay_from_due_date: float = Field(..., examples=[10])
    Num_of_Delayed_Payment: float = Field(..., ge=0, examples=[3])
    Changed_Credit_Limit: float = Field(..., examples=[5.5])
    Num_Credit_Inquiries: float = Field(..., ge=0, examples=[2])
    Outstanding_Debt: float = Field(..., ge=0, examples=[1200.0])
    Credit_Utilization_Ratio: float = Field(..., ge=0, examples=[30.5])
    Credit_History_Age: float = Field(
        ..., ge=0, description="Lama riwayat kredit dalam satuan BULAN", examples=[120]
    )
    Total_EMI_per_month: float = Field(..., ge=0, examples=[150.0])
    Amount_invested_monthly: float = Field(..., ge=0, examples=[200.0])
    Monthly_Balance: float = Field(..., examples=[400.0])

    Occupation: str = Field(..., examples=["Engineer"])
    Credit_Mix: str = Field(..., examples=["Good"])
    Payment_of_Min_Amount: str = Field(..., examples=["Yes"])
    Payment_Behaviour: str = Field(..., examples=["High_spent_Small_value_payments"])


class PredictionOutput(BaseModel):
    prediction: str
    probabilities: Optional[dict] = None
    model_used: str


@app.get("/")
def root():
    return {"message": "Credit Score Classification API aktif", "status": "ok"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_wrapper is not None,
        "model_name": model_wrapper.model_name if model_wrapper else None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(data: CreditScoreInput):
    if model_wrapper is None:
        raise HTTPException(status_code=503, detail="Model belum siap dimuat.")

    try:
        result = model_wrapper.predict(data.model_dump())
        return {
            "prediction": result["prediction"],
            "probabilities": result.get("probabilities"),
            "model_used": model_wrapper.model_name,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal melakukan prediksi: {e}")
