from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable


def _candidate_dirs() -> Iterable[Path]:
    explicit = os.getenv("VH_FFMPEG_BIN", "").strip()
    if explicit:
        yield Path(explicit)

    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        winget_root = Path(local_app_data) / "Microsoft" / "WinGet"
        yield winget_root / "Links"
        packages_root = winget_root / "Packages"
        if packages_root.exists():
            yield from packages_root.glob("Gyan.FFmpeg*/*/bin")

    program_data = os.getenv("ProgramData", "").strip()
    if program_data:
        chocolatey_root = Path(program_data) / "chocolatey" / "lib"
        if chocolatey_root.exists():
            yield from chocolatey_root.glob("ffmpeg*/tools/ffmpeg/bin")

    yield Path("C:/ffmpeg/bin")


def media_executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found

    exe_name = name if name.lower().endswith(".exe") else f"{name}.exe"
    for directory in _candidate_dirs():
        candidate = directory / exe_name
        if candidate.exists():
            return str(candidate)
    return name


def ffmpeg_exe() -> str:
    return media_executable("ffmpeg")


def ffprobe_exe() -> str:
    return media_executable("ffprobe")


def ffmpeg_available() -> bool:
    return Path(ffmpeg_exe()).exists() or shutil.which("ffmpeg") is not None


def ensure_ffmpeg_on_path() -> None:
    for executable in (ffmpeg_exe(), ffprobe_exe()):
        path = Path(executable)
        if not path.exists():
            continue
        directory = str(path.parent)
        current = os.environ.get("PATH", "")
        items = [item for item in current.split(os.pathsep) if item]
        if directory not in items:
            os.environ["PATH"] = os.pathsep.join([directory] + items)
