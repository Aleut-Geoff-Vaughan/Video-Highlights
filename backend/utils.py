from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    value = uuid4().hex
    return f"{prefix}_{value}"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def encode_cursor(offset: int) -> str:
    raw = str(max(0, offset)).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def decode_cursor(cursor: Optional[str]) -> int:
    if not cursor:
        return 0
    padding = "=" * (-len(cursor) % 4)
    decoded = base64.urlsafe_b64decode((cursor + padding).encode("utf-8")).decode("utf-8")
    try:
        value = int(decoded)
    except ValueError:
        return 0
    return max(0, value)
