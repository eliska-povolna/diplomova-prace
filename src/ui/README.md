# UI — Interactive POI Recommender Interface

> **Status: Planned** — the frontend technology has not been decided yet.

## Purpose

The UI provides a web-based interface for:

1. **Browsing recommendations** — display a ranked list of POIs for a given user.
2. **Inspecting explanations** — show the top-activated SAE features that drove each recommendation.
3. **Steering preferences** — expose interactive sliders / knobs tied to the SAE feature dimensions so that users can adjust their preference profile and see recommendations update in real time.

## Candidate technologies

| Option | Notes |
|---|---|
| **Streamlit** | Fastest to prototype; pure Python; good for research demos |
| **Gradio** | Similar to Streamlit; excellent HuggingFace integration |
| **FastAPI + React** | Most flexible; better for a production-quality UI |
| **Dash (Plotly)** | Good for data-heavy dashboards |

A decision will be made once the model architecture stabilises.

## Planned API surface (backend)

Regardless of the frontend choice, the backend will expose at least:

| Endpoint | Description |
|---|---|
| `GET /recommend/{user_id}` | Return top-K recommended POIs |
| `GET /explain/{user_id}` | Return top SAE feature activations for a user |
| `POST /steer` | Accept feature overrides; return re-ranked POIs |

## Getting started (once implemented)

```bash
# Instructions will be added here
```
