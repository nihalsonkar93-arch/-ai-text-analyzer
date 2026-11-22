
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from transformers import pipeline

# Load the Hugging Face sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

# Load the Hugging Face summarization model
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-6-6",
    device=-1
)

# Create the FastAPI application
app = FastAPI(title="AI Text Analyzer")


# Validate incoming text
class TextInput(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Text cannot contain only spaces.")
        return value


# Endpoint 1: Sentiment analysis
@app.post("/analyze")
def analyze_text(data: TextInput):
    prediction = sentiment_model(data.text)[0]

    return {
        "text": data.text,
        "sentiment": prediction["label"],
        "confidence": round(float(prediction["score"]), 3)
    }


# Endpoint 2: Text summarization
@app.post("/summarize")
def summarize_text(data: TextInput):
    summary = summarizer(
        data.text,
        max_length=60,
        min_length=20,
        do_sample=False
    )

    return {
        "original_text": data.text,
        "summary": summary[0]["summary_text"]
    }


# Endpoint 3: Combined analysis
@app.post("/analyze-all")
def analyze_all(data: TextInput):
    prediction = sentiment_model(data.text)[0]

    summary = summarizer(
        data.text,
        max_length=60,
        min_length=20,
        do_sample=False
    )

    return {
        "original_text": data.text,
        "sentiment": prediction["label"],
        "confidence": round(float(prediction["score"]), 3),
        "summary": summary[0]["summary_text"]
    }
