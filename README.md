# Trustworthy Model Registry

## Overview

The **Trustworthy Model Registry (TMR)** is a cloud-native system for ingesting, evaluating, storing, and serving machine learning artifacts (models, datasets, and code) with explicit, machine-verifiable trust signals. It is designed for enterprise environments that require stronger guarantees than public registries such as npm or HuggingFace alone, including reproducibility, licensing compatibility, lineage transparency, and security-aware ingestion.

The registry exposes a strictly **OpenAPI-compliant REST API** and a browser-based UI, and is deployed entirely on AWS using free-tier-compatible components. Automated CI/CD, testing, observability, and security analysis are integral parts of the system.

---

## Links
* **API**: https://c1r52eygxi.execute-api.us-east-2.amazonaws.com/docs
* **Frontend**: http://sot4-model-registry-dev.s3-website.us-east-2.amazonaws.com/

## Summary of Key Capabilities

### Artifact Lifecycle (Baseline)

* **Upload**: Register models, datasets, or code artifacts via URL or direct upload.
* **Ingest**: Pull and vet public HuggingFace models automatically.
* **Rate**: Compute trust metrics and sub-scores for artifacts.
* **Download**: Retrieve full artifacts or selected components (e.g., weights only).
* **Enumerate & Search**: List all artifacts or search using regex over names and model cards.
* **Delete**: Remove artifacts by ID.
* **Reset**: Restore the registry to a clean default state.

### Trust & Governance Features

* **Reproducibility scoring** (0, 0.5, 1.0)
* **Reviewedness** based on GitHub PR review history
* **Lineage graphs** derived from model metadata (e.g., `config.json`)
* **Tree score** aggregation over parent models
* **License compatibility checks** (fine-tune + inference)
* **Size / cost estimation** prior to download

### Extended / Advanced Features (Track-dependent)

* Authentication and role-based access control (Security Track)
* Sensitive model handling with pre-download JavaScript hooks
* Package confusion attack detection
* Performance benchmarking and bottleneck analysis
* High-assurance testing and disaster-proofing

---

## System Architecture

### High-Level Architecture

```
Client (UI / API)
        |
        v
FastAPI Application (REST, OpenAPI)
        |
        +--> Registry Service (metadata, lineage)
        +--> Scoring Service (metrics computation)
        +--> Storage Service (artifacts)
        |
        v
AWS Infrastructure
```

### AWS Components

* **AWS Lambda**: Stateless execution of the API backend
* **Amazon API Gateway**: Public REST interface
* **Amazon S3**: Artifact storage (models, datasets, code, and registry.json)
* **CloudWatch**: Logs, metrics, and system health

All components are selected to remain within AWS Free Tier limits.

---

## API Overview

The system fully complies with the provided OpenAPI specification. Below is a functional summary of major endpoints.

### Artifact Management

* `POST /artifact` – Upload an artifact
* `POST /artifact/ingest` – Ingest a public HuggingFace model
* `GET /artifact/{type}/{id}` – Get artifact by ID
* `GET /artifact/byName/{name}` – Get artifacts by exact name
* `DELETE /artifact/{type}/{id}` – Delete artifact

### Rating & Analysis

* `GET /artifact/{type}/{id}/rate` – Compute and return trust metrics
* `GET /artifact/{type}/{id}/lineage` – Retrieve lineage graph
* `GET /artifact/{type}/{id}/cost` – Size and cost estimation
* `POST /artifact/license-check` – License compatibility analysis

### Enumeration & Search

* `GET /artifacts/{type}` – Enumerate artifacts (paged)
* `GET /artifacts/search` – Regex-based search

### System & Ops

* `GET /health` – System health and recent activity
* `DELETE /reset` – Reset registry to default state

Authentication headers are required where specified by the OpenAPI schema.

---

## CI/CD Pipeline

### Continuous Integration (CI)

Implemented using **GitHub Actions**:

* Triggered on every pull request
* Runs unit, feature, and end-to-end tests
* Enforces minimum coverage thresholds
* Performs linting and static checks

### Continuous Deployment (CD)

* Triggered on merge to the main branch
* Automatically deploys the service to AWS
* Verifies successful startup and health endpoint

### Dependency & Code Quality Tooling

* **Dependabot** for dependency updates
* **GitHub Copilot Auto-Review** for automated PR feedback
* **Microsoft Accesibility Insights** for ADA compliance testing

---

## Security & Reliability

* Designed using **STRIDE threat modeling**
* Risks analyzed against **OWASP Top 10**
* Input validation and strict schema enforcement
* Explicit handling of malicious or non-compliant artifacts
* CloudWatch-based logging and monitoring

Where enabled, authentication tokens expire after a fixed time or number of API calls.

---

## Local Development Setup

### Prerequisites

* Python 3.12
* Git
* AWS account (free tier)

### Installation

```bash
git clone <repository-url>
cd trustworthy-model-registry
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Set required environment variables:

```bash
export AWS_REGION=us-east-2
export STORAGE_BUCKET=<your-s3-bucket>
export AUTH_TOKEN=<default-admin-token>
```

### Running Locally

```bash
uvicorn src.run:app --reload
```

The API will be available at:

```
http://localhost:8000
```

Swagger UI:

```
http://localhost:8000/docs
```

---

## Deployment

Deployment is handled automatically via GitHub Actions. For manual deployment or debugging:

1. Package the application
2. Deploy to AWS Lambda
3. Attach API Gateway routes
4. Verify `/health` endpoint

All deployments assume stateless execution with externalized storage.

---

## Testing

* **Unit Tests**: Component-level logic
* **Feature Tests**: End-to-end API behavior
* **System Tests**: Full workflows (upload → rate → download)

Coverage reports are generated as part of CI.

---

## Observability

* `/health` endpoint exposes recent activity
* CloudWatch logs capture all requests, responses, and errors
* Metrics support performance and reliability analysis

---

## Project Purpose & Value

The Trustworthy Model Registry goes beyond traditional package managers by embedding **trust, compliance, and governance directly into the artifact lifecycle**. Instead of relying on popularity or manual review, engineering teams gain automated, repeatable assurances about the models they deploy—reducing operational risk, legal exposure, and time-to-production.

---

## License

This project is provided for academic and demonstration purposes as part of ECE 461/30861 Software Engineering.
