from __future__ import annotations
from typing import Dict, Any
import os, json
from huggingface_hub import HfApi
from dotenv import load_dotenv
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
# utils used by dataset/code metrics are already in your repo (seen in CURRENT CODE)
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
            config_file = next((s for s in siblings if s.rfilename == "config.json"), None)
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
        if "```" in card:
            resource["demo_code"] = card
        else:
            resource["demo_code"] = ""

        print(json.dumps(resource, indent=2)[:1000])  # debug
        return resource


    # ---------------------------------------------------------------------- #
    # RATE (compute all metrics)
    # ---------------------------------------------------------------------- #
    def rate(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute metrics for a given model resource and return:
        {
          "net": float,
          "subs": {metric_name: score_or_object},
          "latency_ms": int
        }
        """
        results: Dict[str, tuple] = {}

        # ----- Phase 1 + Phase 2 metrics -----
        metric_map = {
            "ramp_up_time": ramp_up_time,
            "bus_factor": bus_factor,
            "performance_claims": performance_claims,
            "license": license_metric,
            "size_score": size_metric,
            "dataset_and_code_score": dataset_and_code_score,
            "dataset_quality": dataset_quality,
            "code_quality": code_quality,
            "reproducibility": reproducibility,
            "reviewedness": reviewedness,
            "treescore": treescore,
        }

        # Collect scores and latencies
        for name, mod in metric_map.items():
            try:
                result = mod.metric(resource)

                if isinstance(result, dict):
                    # --- Phase 1 size metric format ---
                    if name == "size_score" and isinstance(result.get("metric"), dict):
                        raw_sizes = result["metric"]
                        latency = result.get("latency") or 0

                        # Average across devices for netscore
                        avg_score = (
                            sum(v for v in raw_sizes.values() if isinstance(v, (int, float)))
                            / max(len(raw_sizes), 1)
                        )
                        results[name] = (avg_score, latency, raw_sizes)

                    # --- All other metrics with dict outputs ---
                    else:
                        score = result.get("score") or result.get("metric") or 0.0
                        latency = result.get("latency") or result.get("latency_ms") or 0
                        results[name] = (score, latency)

                elif isinstance(result, (list, tuple)) and len(result) >= 2:
                    results[name] = (result[0], result[1])
                else:
                    results[name] = (0.0, 0)

            except Exception as e:
                results[name] = (0.0, 0)
                print(f"[WARN] {name} metric failed: {e}")

        # Compute aggregate (same as Phase 1 CLI executable)
        numeric_scores = []
        net_latency = 0
        for v in results.values():
            score = v[0]
            latency = v[1]
            try:
                net_latency += int(latency or 0)
            except Exception:
                pass
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                numeric_scores.append(float(score))

        net_score = round(
            (sum(numeric_scores) / len(numeric_scores)) if numeric_scores else 0.0, 4
        )

        # Build subs for API response
        subs: Dict[str, Any] = {}
        for k, v in results.items():
            try:
                if k == "size_score":
                    # return the full device map for size_score
                    if len(v) == 3 and isinstance(v[2], dict):
                        subs[k] = v[2]
                    elif isinstance(v[0], dict):
                        subs[k] = v[0]
                    else:
                        subs[k] = float(v[0])
                else:
                    if isinstance(v[0], dict):
                        maybe_val = (
                            v[0].get("score")
                            or v[0].get("metric")
                            or v[0].get("value")
                            or 0.0
                        )
                        subs[k] = maybe_val
                    else:
                        subs[k] = float(v[0])

            except Exception as e:
                subs[k] = 0.0
                print(f"[WARN] Failed to serialize {k}: {e}")

        return {"net": net_score, "subs": subs, "latency_ms": net_latency}
