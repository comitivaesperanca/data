from typing import Union
from entities.predict_content import predict_content
from entities.error_message import error_message
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from fastapi import HTTPException, Response
from models.LIABertClassifier import LIABertClassifier
import torch.nn as nn
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import (
    AutoModelForPreTraining,
)  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
from transformers import BertTokenizer, BertModel
import torch

# Pré-processamento dos dados
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("portuguese"))


def preprocess_text(text):
    # remover acentuação
    text = unidecode(text)
    # Remover pontuações
    text = re.sub(r"[^\w\s]", "", text)
    # Tokenização
    words = word_tokenize(text.lower())
    # Remover stopwords
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


async def predict_text(
    text_content: str, model_type: str, response=Response
) -> Union[str, float]:
    text_processed = preprocess_text(text_content)

    if model_type == "naive_bayes":
        print(text_processed)
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text_processed])
        print(X)
        # # Carregar o modelo
        model = pickle.load(open("models/modelo_naive_soja.pkl", "rb"))

        raise HTTPException(status_code=400, detail="Modelo não implementado.")

    elif model_type == "rede_neural":
        tokenizer = BertTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased", num_labels=3
        )
        model = BertModel.from_pretrained("neuralmind/bert-base-portuguese-cased")

        device = torch.device("cpu")

        # Crie uma instância de seu modelo
        modelo_soja = LIABertClassifier(model, 3)
        modelo_soja.load_state_dict(
            torch.load(
                "models/modelo_redeneural_bert_soja.pt",
                map_location=torch.device("cpu"),
            ),
            strict=False,
        )

        modelo_soja.to(device)

        text_processed_token = tokenizer.encode(text_processed, return_tensors="pt")
        outputs = modelo_soja(text_processed_token)
        final_sentiment = ""
        if outputs.argmax(axis=1).cpu().numpy().tolist()[0] == 0:
            final_sentiment = "Negativa"
        elif outputs.argmax(axis=1).cpu().numpy().tolist()[0] == 1:
            final_sentiment = "Positiva"
        else:
            final_sentiment = "Neutra"

        ## mostre a probabilidade (de 0 a 100%) de cada classe
        probs = outputs.softmax(axis=1).cpu().detach().numpy().tolist()[0]
        return predict_content(
            text_content=text_content,
            model_type=model_type,
            final_sentiment=final_sentiment,
            negative=probs[0] * 100,
            positive=probs[1] * 100,
            neutral=probs[2] * 100,
        )
