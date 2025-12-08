import re
from urllib.parse import urlparse

def normalize_hf_id(raw: str) -> str:
    """
    Normalize any HuggingFace model/dataset URL or identifier into the canonical form:
        owner/model-name

    Accepts inputs like:
        - https://huggingface.co/google-bert/bert-base-uncased
        - huggingface.co/google-bert/bert-base-uncased
        - google-bert/bert-base-uncased/
        - google-bert/bert-base-uncased/tree/main
        - GOOGLE-BERT/BERT-BASE-UNCASED
        - trailing slash, query params, fragments

    Returns:
        'google-bert/bert-base-uncased'
    """

    if not raw:
        return ""

    s = raw.strip()

    # 1. Remove protocol
    s = re.sub(r'^https?://', '', s, flags=re.IGNORECASE)

    # 2. Remove domain prefix (huggingface.co or www.huggingface.co)
    s = re.sub(r'^(www\.)?huggingface\.co/', '', s, flags=re.IGNORECASE)

    # 3. Remove URL query parameters and fragments
    s = s.split("?", 1)[0].split("#", 1)[0]

    # 4. Split path into segments
    parts = [p for p in s.split("/") if p]

    # We expect at least owner/repo; if too short, return as-is
    if len(parts) < 2:
        return s.lower()

    owner = parts[0]
    repo = parts[1]

    # If path contains extras like tree/main or resolve/main — ignore everything after the repo
    # Example: google/model/tree/main --> google/model
    return f"{owner}/{repo}".lower()
