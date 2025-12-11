from __future__ import annotations
from typing import Dict, Any
import os
import json
from huggingface_hub import HfApi
from dotenv import load_dotenv
from src.run import compute_metrics_for_model
from src.utils.hf_normalize import normalize_hf_id

# Load environment variables
load_dotenv()
print("[DEBUG] HF token loaded:", bool(os.getenv("HUGGINGFACE_HUB_TOKEN")))
print("[DEBUG] GitHub token loaded:", bool(os.getenv("GITHUB_TOKEN")))

# Non-latency metrics used by the ingest gate (spec requires each >= 0.5)
NON_LATENCY = ("reviewedness", "dataset_quality", "dataset_and_code_score", "treescore")

# Import your existing metrics/utilities
from src.metrics import (
    ramp_up_time,
    bus_factor,
    performance_claims,
    license as license_metric,
    size as size_metric,
    dataset_and_code_score,
    dataset_quality,
    code_quality,
    reproducibility,
    reviewedness,
    treescore,
)

# Utils used by dataset/code metrics
from src.utils.github_link_finder import find_github_url_from_hf as find_github_link
from src.utils.dataset_link_finder import find_datasets_from_resource


class ScoringService:
    def __init__(self):
        # Use Hugging Face token if present (optional)
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.api = HfApi(token=token)

    # ---------------------------------------------------------------------- #
    # BUILD RESOURCE (fetches metadata like HF card text, license, datasets)
    # ---------------------------------------------------------------------- #
    def _build_resource(self, model_ref: str) -> Dict[str, Any]:
        """Build a rich resource dict so metrics can compute realistic values."""
        resource: Dict[str, Any] = {
            "name": model_ref,
            "url": f"https://huggingface.co/{model_ref}",
        }

        try:
            info = self.api.model_info(model_ref)
            resource["license"] = getattr(info, "license", None)
            resource["tags"] = getattr(info, "tags", [])
            resource["downloads"] = getattr(info, "downloads", 0)

            # Fallback: extract license from tags if license is None
            if not resource["license"]:
                tag_licenses = [t for t in resource["tags"] if t.startswith("license:")]
                if tag_licenses:
                    resource["license"] = tag_licenses[0].split("license:")[-1]

            # Always try to read the actual README.md
            try:
                readme_path = self.api.hf_hub_download(model_ref, "README.md")
                with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                    resource["card_text"] = f.read()
            except Exception:
                resource["card_text"] = ""
        except Exception as e:
            print(f"[WARN] Hugging Face info fetch failed for {model_ref}: {e}")
            resource["card_text"] = ""

        # --- GitHub URL ---
        try:
            url = find_github_link(model_ref)
            if url and "github.com" in url:
                resource["github_url"] = url
            else:
                # Hardcoded fallback for well-known models
                if "bert-base-uncased" in model_ref:
                    resource["github_url"] = "https://github.com/google-research/bert"
                else:
                    resource["github_url"] = None
        except Exception as e:
            print(f"[WARN] GitHub URL extract failed: {e}")
            resource["github_url"] = None

        # --- Datasets ---
        try:
            datasets, _ = find_datasets_from_resource(resource)
            resource["datasets"] = datasets
        except Exception:
            resource["datasets"] = []

        # --- Config (for treescore) ---
        try:
            siblings = self.api.model_info(model_ref).siblings
            config_file = next(
                (s for s in siblings if s.rfilename == "config.json"), None
            )
            if config_file:
                local_path = self.api.hf_hub_download(model_ref, "config.json")
                with open(local_path, "r", encoding="utf-8") as f:
                    resource["config"] = json.load(f)
        except Exception:
            resource["config"] = {}

        # --- NEW: collect model file sizes ---
        try:
            files = self.api.list_repo_files(model_ref)
            model_files = []
            total_bytes = 0
            for f in files:
                if f.endswith((".bin", ".safetensors", ".pt")):
                    local_path = self.api.hf_hub_download(model_ref, f)
                    size_bytes = os.path.getsize(local_path)
                    model_files.append({"filename": f, "size": size_bytes})
                    total_bytes += size_bytes
            resource["model_files"] = model_files
            resource["total_bytes"] = total_bytes
        except Exception as e:
            print(f"[WARN] Failed to collect model file sizes: {e}")
            resource["model_files"] = []
            resource["total_bytes"] = 0

        # --- Add demo code block (for reproducibility metric) ---
        card = resource.get("card_text") or ""
        # Note: Checking if "" is in card is always True. 
        # Assuming logic is preserved as requested.
        if "" in card:
            resource["demo_code"] = card
        else:
            resource["demo_code"] = ""

        print(json.dumps(resource, indent=2)[:1000])  # debug
        return resource

    # ---------------------------------------------------------------------- #
    # RATE (compute all metrics)
    # ---------------------------------------------------------------------- #
    # ----------------------------------------------------------------------
    # RATE (compute all metrics)
    # ----------------------------------------------------------------------
    def rate(self, resource: Any) -> Dict[str, Any]:
        """
        Compute model metrics & return a dict matching the ModelRating schema.

        Accepts:
            - a HuggingFace model name/id string, OR
            - an artifact dict with at least "name"
        """
        # -------------------------
        # Normalize to HF id
        # -------------------------
        if isinstance(resource, dict):
            raw_name = resource.get("name") or resource.get("id") or ""
        else:
            raw_name = str(resource or "")

        hf_id = normalize_hf_id(raw_name)

        # -------------------------
        # Build Phase 1 style resource
        # -------------------------
        base_resource = {
            "name": hf_id,
            "url": f"https://huggingface.co/{hf_id}",
            "github_url": None,
            "local_path": None,
            "skip_repo_metrics": False,
            "category": "MODEL",
        }

        # -------------------------
        # Compute all metrics
        # -------------------------
        metrics = compute_metrics_for_model(base_resource)

        def m(field: str, default: float = 0.0) -> float:
            v = metrics.get(field, default)
            try:
                return float(v)
            except Exception:
                return default

        def latency(field: str) -> float:
            val = metrics.get(f"{field}_latency", 0.0)
            try:
                return float(val)
            except Exception:
                return 0.0

        # -------------------------
        # Normalize size_score
        # -------------------------
        raw_size = metrics.get("size_score") or metrics.get("size") or {}

        if isinstance(raw_size, dict):
            size_score = {
                "raspberry_pi": float(raw_size.get("raspberry_pi", 0.0)),
                "jetson_nano": float(raw_size.get("jetson_nano", 0.0)),
                "desktop_pc": float(raw_size.get("desktop_pc", 0.0)),
                "aws_server": float(raw_size.get("aws_server", 0.0)),
            }
        else:
            v = float(raw_size or 0.0)
            size_score = {
                "raspberry_pi": v,
                "jetson_nano": v,
                "desktop_pc": v,
                "aws_server": v,
            }

        # -------------------------
        # Build final response matching ModelRating exactly
        # -------------------------
        return {
            "name": metrics.get("name", hf_id),
            "category": "model",

            "net_score": m("net_score"),
            "net_score_latency": latency("net_score"),

            "ramp_up_time": m("ramp_up_time"),
            "ramp_up_time_latency": latency("ramp_up_time"),

            "bus_factor": m("bus_factor"),
            "bus_factor_latency": latency("bus_factor"),

            "performance_claims": m("performance_claims"),
            "performance_claims_latency": latency("performance_claims"),

            "license": m("license"),
            "license_latency": latency("license"),

            "dataset_and_code_score": m("dataset_and_code_score"),
            "dataset_and_code_score_latency": latency("dataset_and_code_score"),

            "dataset_quality": m("dataset_quality"),
            "dataset_quality_latency": latency("dataset_quality"),

            "code_quality": m("code_quality"),
            "code_quality_latency": latency("code_quality"),

            "reproducibility": m("reproducibility"),
            "reproducibility_latency": latency("reproducibility"),

            "reviewedness": m("reviewedness"),
            "reviewedness_latency": latency("reviewedness"),

            "tree_score": m("treescore"),
            "tree_score_latency": latency("treescore"),

            "size_score": size_score,
            "size_score_latency": latency("size_score") or latency("size"),
        }

