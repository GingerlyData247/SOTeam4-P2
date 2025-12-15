# SOTeam4 – Model Registry & Scoring API (Phase 2)

ECE 461 – Software Engineering (Fall 2025)

## Team Members
- Wesley Cameron Todd
- Esau Cortez
- Ethan Surber
- Sam Brahim

---

## Overview
This project implements a **model registry and automated evaluation service** for machine learning models, designed for the ECE 461 Phase 2 final deliverable. The system allows users to ingest ML models (primarily from Hugging Face), compute a standardized set of quality metrics, store results in a registry, and retrieve ratings and artifacts via a REST API and web frontend.

The goal is to help organizations quickly assess whether a model is **usable, reliable, reproducible, and legally compatible** before adoption.

The system consists of:
- A **FastAPI backend** deployed on AWS (Lambda + API Gateway)
- **S3-backed storage** for registry state and model artifacts
- A **web frontend** for interactive usage
- A **metrics engine** that computes all required Phase 2 scores

---

## Deployed Endpoints

### API
```
https://c1r52eygxi.execute-api.us-east-2.amazonaws.com
```

### Web UI
```
http://sot4-model-registry-dev.s3-website.us-east-2.amazonaws.com/
```

---

## Core Features

### 1. Artifact Registry
- Persistent registry stored in **S3** (`registry/registry.json`)
- Supports create, retrieve, delete, enumerate, and count operations
- Metadata preserved exactly as required by the Phase 2 spec
- Hardened against malformed registry entries to avoid runtime failures

### 2. Model Ingestion
- Hugging Face model ingestion via URL
- Normalization of HF IDs
- Automatic metric computation during ingest
- Lineage extraction (parent models) for tree score computation

### 3. Automated Model Rating
- Computes all required Phase 2 metrics:
  - Net score
  - Ramp-up time
  - Bus factor
  - Performance claims
  - License suitability
  - Dataset & code score
  - Dataset quality
  - Code quality
  - Reproducibility
  - Reviewedness
  - Tree score
  - Size score (Raspberry Pi, Jetson Nano, Desktop PC, AWS Server)
- Latency recorded per metric
- Output strictly matches the `ModelRating` OpenAPI schema

### 4. Artifact Download
- Secure artifact download via **presigned S3 URLs**
- Supports full artifact downloads
- Returns correct HTTP status codes for rejected, missing, or malformed requests

### 5. Cloud-Ready Architecture
- **FastAPI** backend
- **Mangum** adapter for AWS Lambda
- **S3** for artifact storage and registry persistence
- **CloudWatch-friendly request logging** via custom ASGI middleware
- Local filesystem fallback for development and testing

---

## API Routes (Summary)

All API routes are prefixed with `/api` unless noted.

### Health & Admin
- `GET /health` – Service health and uptime
- `DELETE /reset` – Reset registry and scoring state
- `GET /tracks` – Returns supported tracks

### Artifact Queries
- `POST /artifact/byRegEx` – Regex search over artifacts
- `GET /artifact/byName/{name}` – Exact name lookup

### Model Operations
- `POST /artifact/model` – Ingest a model
- `GET /artifact/model/{id}/rate` – Retrieve model rating
- `GET /artifact/model/{id}/download` – Download model artifact

### S3 Utility Routes
- `POST /api/s3/put-text`
- `GET /api/s3/get-text`
- `POST /api/s3/upload`

---

## API Baseline Endpoints

### Registry & Artifact APIs
- `POST /api/artifact/{artifact_type}` – Upload artifact
- `GET /api/artifact/{artifact_type}` – Enumerate artifacts
- `GET /api/artifact/{artifact_type}/{id}` – Retrieve artifact metadata
- `DELETE /api/artifact/{artifact_type}/{id}` – Delete artifact

### Rating
- `GET /api/artifact/{artifact_type}/rate` – Compute or retrieve model rating

### Download
- `GET /api/artifact/{artifact_type}/download` – Download artifact

### Utility
- `GET /env` – Environment diagnostics (debug only)

All baseline endpoints are implemented to **100% completeness** per the Phase 2 specification and validated via automated autograder tests and manual AWS deployment testing.

---

## Architecture

```
Client (Browser)
   │
   ▼
Frontend (S3 Static Site)
   │
   ▼
API Gateway
   │
   ▼
AWS Lambda (FastAPI + Mangum)
   │
   ├── Metrics Engine
   ├── Registry Service (S3)
   ├── Storage Service (S3 / Local)
   └── Hugging Face + GitHub APIs
```

---

## Environment Variables

The following environment variables are required or supported:

```bash
# AWS
S3_BUCKET=<bucket-name>
AWS_REGION=us-east-2

# Optional tokens (strongly recommended)
HUGGINGFACE_HUB_TOKEN=<token>
GITHUB_TOKEN=<token>

# Local development
LOCAL_STORAGE=1
```

---

## Local Development

### Requirements
- Python 3.12+
- AWS credentials (for S3 mode)

### Install
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
uvicorn src.main:app --reload
```

If `LOCAL_STORAGE=1` is set, artifacts are written to `/tmp/local-artifacts` instead of S3.

---

## Deployment

- Designed for AWS Lambda + API Gateway
- Uses S3 for registry persistence and artifact storage
- CORS configured explicitly in the backend (API Gateway CORS disabled)

---

## Validation & Testing

- **Automated end-to-end tests** (autograder)
- **Manual AWS deployment validation** using CloudWatch logs
- **Negative testing** for malformed inputs, unsupported artifact types, and missing resources
- Consistent HTTP status code handling per instructor clarification

---

## Notes on Phase 2 Compliance

- License metric returns a **numeric suitability score**, not a string
- Size score is a structured object with four deployment targets
- Reviewedness gating is enforced where required by the spec
- Tree score and lineage are extracted from HF metadata and configs
- All required OpenAPI fields are populated

---

## License

This project is developed for academic use as part of ECE 461 at Purdue University.

---

## Acknowledgements
- Hugging Face Hub
- GitHub API
- AWS (Lambda, S3, CloudWatch)
- FastAPI & Mangum
