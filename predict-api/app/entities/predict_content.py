from pydantic import BaseModel

class predict_content(BaseModel):
    text_content: str
    model_type: str
    final_sentiment: str
    positive: float
    negative: float
    neutral: float