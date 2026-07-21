from pathlib import Path
import logging
import tempfile
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from utils.model import predict_probability, predict
from utils.preprocessing import prepare_features
from utils.risk import generate_risk_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Banking Fraud Detection API",
    description="""
REST API for Banking Fraud Detection

Features
- Single Transaction Prediction
- Batch CSV Prediction
- Fraud Probability
- Risk Scoring
- Confidence Score
""",
    version="1.0.0",
)


class TransactionRequest(BaseModel):
    data: dict


@app.get("/", tags=["Home"])
def home():

    return {

        "Project": "Banking Fraud Detection",

        "Version": "1.0.0",

        "Status": "Running",

        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "Available Endpoints": [
            "/health",
            "/model-info",
            "/predict",
            "/batch-predict"
        ]
    }


@app.get("/health", tags=["Health"])
def health():

    return {

        "Status": "Healthy",

        "Model Loaded": True,

        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    }


@app.get("/model-info", tags=["Model"])
def model_info():

    return {
        "Model": "Best Trained Model",
        "Task": "Binary Fraud Classification",
        "Output": [
            "Prediction",
            "Fraud Probability",
            "Risk Level",
            "Confidence",
            "Recommended Action"
        ]
    }


@app.post("/predict", tags=["Prediction"])
def predict_transaction(request: TransactionRequest):

    try:

        df = pd.DataFrame([request.data])

        df = prepare_features(df)

        prediction = int(
            predict(df)[0]
        )

        probability = float(
            predict_probability(df)[0]
        )

        report = generate_risk_report(
            probability,
            prediction
        )

        logger.info("Single prediction completed.")

        return {

        "Success": True,

        "Prediction Result": report,

        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        }

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/batch-predict", tags=["Batch Prediction"])
async def batch_predict(file: UploadFile = File(...)):

    try:

        if not file.filename.endswith(".csv"):

            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported."
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv"
        ) as tmp:

            tmp.write(await file.read())

            temp_path = tmp.name

        df = pd.read_csv(temp_path)

        processed = prepare_features(df)

        predictions = predict(processed)

        probabilities = predict_probability(processed)

        reports = []

        for pred, prob in zip(predictions, probabilities):

            reports.append(
                generate_risk_report(
                    float(prob),
                    int(pred)
                )
            )

        result = df.copy()

        result["Prediction"] = [
            r["Prediction"]
            for r in reports
        ]

        result["Fraud Probability"] = [
            r["Fraud Probability"]
            for r in reports
        ]

        result["Risk Level"] = [
            r["Risk Level"]
            for r in reports
        ]

        result["Confidence"] = [
            r["Confidence"]
            for r in reports
        ]

        result["Confidence Level"] = [
            r["Confidence Level"]
            for r in reports
        ]

        result["Recommended Action"] = [
            r["Recommended Action"]
            for r in reports
        ]

        output_dir = BASE_DIR / "data" / "processed"

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = output_dir / "batch_predictions.csv"

        result.to_csv(
            output_file,
            index=False
        )

        logger.info(
            f"{len(result)} transactions processed."
        )

        return {

        "Success": True,

        "Rows Processed": len(result),

        "Output File": str(output_file),

        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        }

    except Exception as e:

        logger.exception(e)

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):

    logger.exception(exc)

    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal Server Error",
            "details": str(exc)
        },
    )