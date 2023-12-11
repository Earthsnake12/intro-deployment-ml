from fastapi import FastAPI
from .app.model import PredictionRequest, PredictionResponse
from .app.view import getPrediction

app = FastAPI(docs_url="/")

@app.post("/v1/predic")
def make_model_predic(request : PredictionRequest):
    return PredictionResponse(worldwide_gross = getPrediction(request))