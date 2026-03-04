# API Specification

**Version**: 0.3.0
**Last Updated**: 2026-03-04

## Base URL
`http://localhost:8000`

## Endpoints

### POST /api/ask
Receive a financial question and return an answer from the LangGraph-powered QA agent (batch response).
The agent parses the question, fetches data from relevant tools (market data, news, knowledge base), and synthesizes a final answer via an LLM call.

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

### POST /api/ask/stream
Stream agent pipeline trace events as **Server-Sent Events (SSE)**. The frontend uses this endpoint for progressive trace updates.

**Request Body**: Same as `/api/ask`.

**Response**: `text/event-stream` — a stream of SSE-formatted events.

Each event has the format:
```
event: <type>
data: <json>

```

#### SSE Event Types

| Event | Payload | Description |
|---|---|---|
| `trace` | `{stage, status, detail, data?}` | Agent pipeline stage update (started/completed) |
| `tool_input` | `{tool, input}` | Input sent to a tool (market_data, news_search, knowledge_base) |
| `tool_output` | `{tool, output}` | Output received from a tool |
| `answer` | `{answer}` | Final synthesized answer |
| `error` | `{message}` | Error during processing |
| `done` | `{}` | Stream complete (always sent last) |

**Example stream**:
```
event: trace
data: {"stage": "parse", "status": "started", "detail": "Extracting entities..."}

event: trace
data: {"stage": "parse", "status": "completed", "detail": "Extracted 1 ticker(s): AAPL"}

event: tool_input
data: {"tool": "market_data", "input": {"tickers": ["AAPL"], "time_period": "5d"}}

event: tool_output
data: {"tool": "market_data", "output": "=== Apple Inc. (AAPL) ===\nCurrent Price: $150.00"}

event: answer
data: {"answer": "AAPL is currently trading at $150.00."}

event: done
data: {}
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
`POST /api/ask` responses follow a consistent envelope:
```json
{
  "status": "ok" | "error",
  "data": <payload or null>,
  "message": "human-readable message"
}
```

`POST /api/ask/stream` returns an SSE stream (see above).
