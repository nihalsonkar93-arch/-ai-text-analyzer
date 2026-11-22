
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from transformers import pipeline
from groq import Groq


# Load the Hugging Face sentiment model
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)


# Connect to Groq securely
groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = (
    Groq(api_key=groq_api_key)
    if groq_api_key else None
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


# Generate summary using Groq
def generate_summary(text):

    if groq_client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured."
        )

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the given text clearly and concisely."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            max_completion_tokens=200
        )

        return response.choices[0].message.content

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Groq summarization request failed."
        )


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

    summary = generate_summary(data.text)

    return {
        "original_text": data.text,
        "summary": summary
    }


# Endpoint 3: Combined analysis
@app.post("/analyze-all")
def analyze_all(data: TextInput):

    prediction = sentiment_model(data.text)[0]
    summary = generate_summary(data.text)

    return {
        "original_text": data.text,
        "sentiment": prediction["label"],
        "confidence": round(float(prediction["score"]), 3),
        "summary": summary
    }

