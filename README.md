
AI Text Analyzer

An AI-powered text analysis application built using Python, FastAPI, and Groq API.

The application provides sentiment analysis, text summarization, and combined text analysis through REST API endpoints.

Features

- Sentiment Analysis: Identifies whether text expresses positive or negative sentiment.
- Text Summarization: Generates concise summaries of longer text.
- Combined Analysis: Performs sentiment analysis and summarization together.
- REST API: Provides interactive API documentation through FastAPI.

Technologies Used

- Python
- FastAPI
- Groq API
- Large Language Models (LLMs)
- Pydantic
- Uvicorn
- GitHub
- Render

API Endpoints

Method| Endpoint| Purpose
POST| "/analyze"| Analyze text sentiment
POST| "/summarize"| Generate a text summary
POST| "/analyze-all"| Perform both operations

Installation and Local Setup

1. Clone the repository:
   
   git clone https://github.com/nihalsonkar93-arch/ml-prediction-api.git

2. Enter the project directory:
   
   cd ai-text-analyzer

3. Install dependencies:
   
   pip install -r requirements.txt

4. Set your Groq API key as an environment variable:
   
   GROQ_API_KEY=your_api_key

5. Start the application:
   
   uvicorn app:app --reload

6. Open the interactive API documentation:
   
   http://127.0.0.1:8000/docs

Deployment

The application is deployed using Render.

Add "GROQ_API_KEY" as an environment variable in the Render dashboard. Never commit API keys or other secrets to GitHub.

What I Learned

- Building REST APIs using FastAPI.
- Integrating external LLM APIs.
- Designing prompts for AI-powered tasks.
- Handling structured JSON responses.
- Validating input using Pydantic.
- Testing API endpoints.
- Deploying an AI application to the cloud.

Project Status

Successfully deployed and tested with all three API endpoints.
