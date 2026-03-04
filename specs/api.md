# API Specification

**Version**: 0.1.0
**Last Updated**: 2026-03-04

## Base URL
`http://localhost:8000`

## Endpoints

### POST /api/ask
Receive a financial question and return an answer from the QA agent.

**Request Body**:
```json
{
  "question": "string (required) — the user's financial question"
}
```

**Success Response** (200):
```json
{
  "status": "ok",
  "data": {
    "question": "What is compound interest?",
    "answer": "Compound interest is..."
  },
  "message": "Question answered successfully"
}
```

**Error Response** (400/500):
```json
{
  "status": "error",
  "data": null,
  "message": "Description of the error"
}
```

### GET /health
Health check endpoint.

**Response** (200):
```json
{
  "status": "ok"
}
```

## Response Format
All API responses follow a consistent envelope:
```json
{
  "status": "ok" | "error",
  "data": <payload or null>,
  "message": "human-readable message"
}
```
