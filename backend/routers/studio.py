"""Studio endpoints: the processing-run library and safe file streaming.

Powers the built-in web UI (served at ``/``): lists completed processing
runs found under the configured output root and streams their artifacts
(videos, manifests, card crops) with path-traversal protection.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ..config import settings

router = APIRouter(tags=["studio"])

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_ALLOWED_SUFFIXES = {".mp4", ".png", ".jpg", ".json", ".csv", ".jsonl", ".log"}


def _output_root() -> Path:
    return Path(settings.output_root).expanduser().resolve()


def _run_dir(run_id: str) -> Path:
    if not _SAFE_NAME.match(run_id):
        raise HTTPException(status_code=400, detail="Invalid run id")
    path = (_output_root() / run_id).resolve()
    if not str(path).startswith(str(_output_root())) or not path.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return path


def _read_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_summary(path: Path) -> Dict[str, object]:
    manifest = _read_json(path / "analysis_bookmarks.json")
    states = _read_json(path / "analysis_game_states.json")
    stats = manifest.get("stats", {}) if isinstance(manifest.get("stats"), dict) else {}
    videos: Dict[str, object] = {}
    if (path / "trimmed_working_video.mp4").exists():
        videos["original"] = "trimmed_working_video.mp4"
    if (path / "debug_camera_wide.mp4").exists():
        videos["debug"] = "debug_camera_wide.mp4"
    for full in sorted(path.glob("full_*_zoom.mp4")):
        videos["zoom"] = full.name
        break
    if (path / "highlights_reel.mp4").exists():
        videos["reel"] = "highlights_reel.mp4"
    if (path / "highlights_montage.mp4").exists():
        videos["montage"] = "highlights_montage.mp4"
    videos["clips"] = sorted(
        f.name for f in path.glob("highlight_*.mp4") if "spotlight" not in f.name
    )
    crops = sorted(f.name for f in (path / "card_crops").glob("*.png")) if (path / "card_crops").is_dir() else []
    return {
        "run_id": path.name,
        "generated_at": manifest.get("generated_at"),
        "video_path": manifest.get("video_path"),
        "stats": stats,
        "state_summary_s": states.get("state_summary_s", {}),
        "goal_events": states.get("goal_events", []),
        "card_events": states.get("card_events", []),
        "set_piece_events": states.get("set_piece_events", []),
        "videos": videos,
        "card_crops": crops,
        "bookmarks": manifest.get("bookmarks", []),
    }


@router.get("/studio/runs")
def list_runs() -> Dict[str, object]:
    root = _output_root()
    runs: List[Dict[str, object]] = []
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if child.is_dir() and (child / "analysis_bookmarks.json").exists():
                summary = _run_summary(child)
                summary.pop("bookmarks", None)  # keep the list light
                runs.append(summary)
            if len(runs) >= 200:
                break
    return {"output_root": str(root), "runs": runs}


@router.get("/studio/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, object]:
    return _run_summary(_run_dir(run_id))


@router.get("/studio/runs/{run_id}/file/{name}")
def get_run_file(run_id: str, name: str) -> FileResponse:
    run = _run_dir(run_id)
    if not _SAFE_NAME.match(name) or Path(name).suffix.lower() not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Invalid file name")
    path = run / name
    if not path.is_file():
        # card crops live one level down
        path = run / "card_crops" / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
    media = "video/mp4" if path.suffix == ".mp4" else None
    return FileResponse(str(path), media_type=media, filename=path.name)
