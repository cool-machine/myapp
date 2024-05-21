from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertModel
from app.models.sentiment_model import predict_sentiment, LSTMModel

from app.preprocessing.clean_text import clean_text
router = APIRouter()

# Load tokenizer and models
tokenizer = BertTokenizer.from_pretrained('models/embedding_bert/tokenizer/')
bert_model = BertModel.from_pretrained('models/embedding_bert/model/')
lstm_model = LSTMModel(input_size=768, hidden_size=256, num_layers=2, num_classes=2, dropout=0.5)
lstm_model.load_state_dict(torch.load('models/predict_bert/lstm_model_state_dict.pth', map_location=torch.device('cpu')))
lstm_model.eval()

class TextRequest(BaseModel):
    text: str

@router.post("/analyze")
async def analyze_sentiment(request: TextRequest):
    try:

        text = request.text
        text = clean_text(text)
        sentiment = predict_sentiment(tokenizer, text, bert_model, lstm_model)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")
