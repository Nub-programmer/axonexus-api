# AxonNexus API

## Overview
AxonNexus API is an AI API gateway that acts as a proxy between users and external AI providers. It provides a unified interface for interacting with various AI models.

## Project Structure
```
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # App configuration and settings
│   │   ├── schemas.py       # Pydantic models for validation
│   │   └── auth.py          # API key authentication dependency
│   └── routes/
│       ├── __init__.py
│       ├── health.py        # Health check endpoint
│       └── chat.py          # Chat completion endpoint
├── requirements.txt
└── .gitignore
```

## API Endpoints

### GET /
Root endpoint returning welcome message and API info.

### GET /health
Health check endpoint returning service status.

### POST /v1/chat (Protected)
OpenAI-compatible chat completion endpoint. **Requires authentication.**

**Header:** `Authorization: Bearer axn_test_123`

Accepts:
```json
{
  "model": "string",
  "messages": [{ "role": "user", "content": "string" }]
}
```

Currently returns mock responses. Ready to be extended for real AI provider integration.

## Authentication
- API key authentication via Bearer token
- Protected endpoints: `/v1/chat`
- Public endpoints: `/`, `/health`, `/docs`
- Default test key: `axn_test_123` (configurable via `API_KEY` env var)

## Running the Application
The app runs on port 5000 using uvicorn.

## Recent Changes
- 2026-01-29: Added API key authentication for /v1/chat endpoint
- 2026-01-29: Initial project setup with FastAPI backend structure

## User Preferences
- Production-style code organization
- Beginner-friendly but structured like a real backend
- Pydantic for request/response validation
