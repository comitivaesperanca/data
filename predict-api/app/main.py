from typing import Union

from fastapi import FastAPI, status, Response
from pydantic import BaseModel
from entities.predict_content import predict_content
from entities.error_message import error_message
from repositories.predict import predict_text
from fastapi.middleware.cors import CORSMiddleware

import urllib.request
import os

model_url = "https://huggingface.co/mfelipemota/comitivaesperanca-soja-model/resolve/main/modelo_soja.pt"
model_path = "models/modelo_redeneural_bert_soja.pt"


# tags
tags_metadata = [
    {
        "name": "predict",
        "description": "Predições realizadas pelos modelos",
    },
]

app = FastAPI(
    title="comitivaesperanca - API de predição de sentimento de textos",
    description="API para realizar predições de sentimento de textos na plataforma [Radar da Soja](https://radar-da-soja.vercel.app/)",
    version="0.0.1",
    openapi_tags=tags_metadata,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.on_event("startup")
async def startup_event():
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)


@app.get(
    "/predict",
    tags=["predict"],
    summary="Predict de texto",
    description="""Endpoint para realizar predições
        utilizando modelo Naive Bayes  ou Rede Neural (baseado em BERT) para 
        realizar o predict do sentimento do textículo encaminhado como parâmetro.""",
    response_description="Predict news category",
    response_model=Union[predict_content, error_message],
)
async def predict(
    text_content: str, model_type: str = "naive_bayes"
) -> Union[str, float]:
    if text_content is None:
        return {"message": "É necessário enviar um texto para realizar a predição."}
    if model_type is None:
        return {
            "message": "É necessário enviar um modelo para realizar a predição. Utilize naive-bayes ou rede-neural"
        }
    if model_type in ["naive_bayes", "rede_neural"]:
        return await predict_text(text_content=text_content, model_type=model_type)
    else:
        # return status code 401
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {
            "message": "É necessário enviar um modelo para realizar a predição. Utilize naive-bayes ou rede-neural"
        }


@app.get(
    "/metrics",
    tags=["predict"],
    summary="Métricas",
    description="""Endpoint para realizar métricas da API.""",
    response_description="Métricas",
    response_model=str,
)
async def metrics() -> str:
    return "Métricas da API de predição de sentimento de textos"


@app.get(
    "/healthcheck",
    tags=["predict"],
    summary="Healthcheck",
    description="""Endpoint para realizar healthcheck da API.""",
    response_description="Healthcheck",
    response_model=str,
)
async def healthcheck() -> str:
    return "API de predição de sentimento de textos"
