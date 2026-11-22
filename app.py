

import os
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from groq import Groq


# Connect to Groq
groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key) if groq_api_key else None

MODEL = "openai/gpt-oss-20b"

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


# Send requests to Groq
def ask_groq(messages, max_tokens=200, json_mode=False):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured."
        )

    try:
        options = {}

        if json_mode:
            options["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=max_tokens,
            **options
        )

        result = response.choices[0].message.content

        if not result:
            raise ValueError("Empty response from Groq.")

        return result

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Groq API request failed."
        )


# Sentiment analysis
def analyze_sentiment(text):

    result = ask_groq(
        [
            {
                "role": "system",
                "content": (
                    "Analyze the sentiment of the given text. "
                    "Return valid JSON with three fields: "
                    "sentiment (POSITIVE or NEGATIVE), "
                    "confidence (a number between 0 and 1), "
                    "explanation (a short reason). "
                    "Confidence is an estimate, not a calibrated probability."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_tokens=150,
        json_mode=True
    )

    try:
        data = json.loads(result)

        sentiment = data["sentiment"].upper()
        confidence = float(data["confidence"])

        if sentiment not in ["POSITIVE", "NEGATIVE"]:
            raise ValueError("Invalid sentiment label.")

        if not 0 <= confidence <= 1:
            raise ValueError("Invalid confidence value.")

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 3),
            "explanation": data.get("explanation", "")
        }

    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        raise HTTPException(
            status_code=502,
            detail="Groq returned an invalid sentiment result."
        )


# Text summarization
def generate_summary(text):

    return ask_groq(
        [
            {
                "role": "system",
                "content": "Summarize the given text clearly and concisely."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        max_tokens=200
    )


# Endpoint 1: Sentiment analysis
@app.post("/analyze")
def analyze_text(data: TextInput):

    prediction = analyze_sentiment(data.text)

    return {
        "text": data.text,
        **prediction
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

    prediction = analyze_sentiment(data.text)
    summary = generate_summary(data.text)

    return {
        "original_text": data.text,
        **prediction,
        "summary": summary
    }

