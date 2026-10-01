from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.scoring import score_churn_customer, score_delivery_order

app = FastAPI(
    title="Zomato BI & ML Scoring API",
    version="1.0.0",
    description="Portfolio API for delivery-time and 60-day churn model scoring.",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict/delivery/{order_id}")
def predict_delivery(order_id: int):
    try:
        return score_delivery_order(order_id)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run python run_pipeline.py first.")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/predict/churn/{customer_id}")
def predict_churn(customer_id: int):
    try:
        return score_churn_customer(customer_id)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifacts not found. Run python run_pipeline.py first.")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
