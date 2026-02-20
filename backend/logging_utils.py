from __future__ import annotations

import logging

from .config import settings

_LOGGING_CONFIGURED = False


def configure_runtime_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    level_name = (settings.log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("uvicorn.access").setLevel(max(logging.INFO, level))
    _LOGGING_CONFIGURED = True
