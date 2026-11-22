
# AI Text Analyzer

An AI-powered text analysis API built using Python, Hugging Face Transformers, and FastAPI.

## Project Overview

This project analyzes text using two pretrained Hugging Face models. It can identify the sentiment of text, summarize it, and return both results through a REST API.

## Features

- Sentiment analysis: identifies positive or negative sentiment.
- Text summarization: generates a shorter version of the input.
- Combined analysis: returns sentiment and summary together.
- Input validation: rejects empty or whitespace-only text.
- JSON responses through FastAPI.

## Technologies Used

- Python
- Hugging Face Transformers
- DistilBERT
- DistilBART
- FastAPI
- Pydantic
- PyTorch

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/analyze` | Sentiment analysis |
| POST | `/summarize` | Text summarization |
| POST | `/analyze-all` | Both operations |

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --reload
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

## How It Works

1. The user submits text to the API.
2. FastAPI validates the input.
3. Hugging Face models analyze or summarize the text.
4. The API returns the results in JSON format.

## Note

This project uses pretrained models for educational and demonstration purposes.
