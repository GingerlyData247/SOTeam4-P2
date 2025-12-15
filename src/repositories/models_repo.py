# SWE 45000, PIN FALL 2025
# TEAM 4
# PHASE 2 PROJECT

# COMPONENT: IN-MEMORY MODEL REPOSITORY
# REQUIREMENTS SATISFIED: artifact persistence abstraction for registry services

# DISCLAIMER: This file contains code either partially or entirely written by
# Artificial Intelligence
"""
src/repositories/models_repo.py

Defines an in-memory repository for storing and managing artifact records.

This module provides a lightweight persistence abstraction used by the
registry service during development, testing, and Phase 2 deployment.
It supports basic CRUD-style operations, simple regex-based listing, and
cursor-based pagination without relying on an external database.

Key responsibilities:
    - Create, retrieve, update, and archive artifact records
    - Maintain artifacts in memory using unique identifiers
    - Support regex-based filtering over artifact names and card text
    - Provide simple cursor-based pagination for list operations
    - Enable fast reset and count operations for testing and system reset

This repository is intentionally simple and deterministic, making it
suitable for unit testing, autograder evaluation, and local execution.
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import re
import uuid

class InMemoryRepo:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def create(self, item: Dict[str, Any]) -> Dict[str, Any]:
        item_id = str(uuid.uuid4())
        item["id"] = item_id
        self._store[item_id] = item
        return item

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(item_id)

    def update(self, item_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if item_id not in self._store:
            return None
        self._store[item_id].update(fields)
        return self._store[item_id]

    def archive(self, item_id: str) -> bool:
        return self._store.pop(item_id, None) is not None

    def list(self, regex: Optional[str], limit: int, cursor: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        # naive cursor as last seen id; OK for baseline in-memory
        ids = list(self._store.keys())
        start = ids.index(cursor) + 1 if cursor in ids else 0
        filtered = []
        pat = re.compile(regex, re.I) if regex else None
        for i in range(start, len(ids)):
            x = self._store[ids[i]]
            if not pat or pat.search(x.get("name","")) or pat.search(x.get("metadata",{}).get("card","")):
                filtered.append(x)
            if len(filtered) == limit:
                next_cur = ids[i]
                return filtered, next_cur
        return filtered, None

    def reset(self):
        self._store.clear()
    
    def count(self) -> int:
        return len(self._store)
