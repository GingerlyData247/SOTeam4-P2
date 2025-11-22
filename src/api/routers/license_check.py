from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from urllib.parse import urlparse
import requests

from src.api.routers.models import _registry  # <-- This is the fix!
from src.api.internal.license import get_license_for_model


router = APIRouter()

class LicenseCheckRequest(BaseModel):
    github_url: str

LICENSE_COMPATIBILITY = {
    "apache-2.0": {"mit", "bsd-3-clause", "bsd-2-clause", "apache-2.0"},
    "mit": {"mit", "bsd-3-clause", "bsd-2-clause"},
    "bsd-3-clause": {"bsd-3-clause", "mit"},
    "bsd-2-clause": {"bsd-2-clause", "mit"},
    "gpl-3.0": set(),
    "cc-by-4.0": set(),
}


def extract_repo_info(url: str):
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub URL. Expected: https://github.com/<owner>/<repo>")
    return parts[0], parts[1]


def fetch_github_license(owner: str, repo: str):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/license"
    resp = requests.get(api_url, headers={"Accept": "application/vnd.github+json"})
    if resp.status_code == 200:
        return resp.json()["license"]["spdx_id"].lower()
    raise ValueError("Unable to determine GitHub license using GitHub API.")


@router.post("/models/{model_id}/license-check")
async def license_check(model_id: str, request: LicenseCheckRequest):
    # 1. Lookup model entry by UUID
    model_entry = _registry.get(model_id)
    if not model_entry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry.")

    model_name = model_entry["name"]

    # 2. Get license from HuggingFace
    try:
        model_license = get_license_for_model(model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model license: {e}")

    # 3. Get GitHub license
    try:
        owner, repo = extract_repo_info(request.github_url)
        github_license = fetch_github_license(owner, repo)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 4. Compare
    compatible = github_license in LICENSE_COMPATIBILITY.get(model_license, set())

    reason = (
        f"{github_license.upper()} is compatible with {model_license.upper()}."
        if compatible
        else f"{github_license.upper()} is NOT compatible with {model_license.upper()}."
    )

    return {
        "model_id": model_id,
        "model_name": model_name,
        "model_license": model_license,
        "github_license": github_license,
        "compatible": compatible,
        "reason": reason
    }
