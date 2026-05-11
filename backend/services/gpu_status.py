from __future__ import annotations

import subprocess
from typing import Any, Dict, List

from .ffmpeg_tools import ffmpeg_exe


def _run_nvidia_smi() -> Dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi was not found on PATH", "gpus": []}
    except Exception as exc:
        return {"available": False, "error": str(exc), "gpus": []}

    if result.returncode != 0:
        return {"available": False, "error": (result.stderr or result.stdout or "").strip(), "gpus": []}

    gpus: List[Dict[str, Any]] = []
    for index, line in enumerate(result.stdout.splitlines()):
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        name, driver, memory_total, memory_used, utilization, temperature = parts[:6]
        gpus.append(
            {
                "index": index,
                "name": name,
                "driver_version": driver,
                "memory_total_mb": _safe_int(memory_total),
                "memory_used_mb": _safe_int(memory_used),
                "utilization_gpu_percent": _safe_int(utilization),
                "temperature_c": _safe_int(temperature),
            }
        )
    return {"available": bool(gpus), "error": None, "gpus": gpus}


def _safe_int(value: object) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _check_ffmpeg_nvenc() -> Dict[str, Any]:
    cmd = [ffmpeg_exe(), "-hide_banner", "-encoders"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=5)
    except FileNotFoundError:
        return {"available": False, "encoder": "h264_nvenc", "error": "ffmpeg was not found on PATH"}
    except Exception as exc:
        return {"available": False, "encoder": "h264_nvenc", "error": str(exc)}

    output = f"{result.stdout}\n{result.stderr}"
    available = result.returncode == 0 and (" h264_nvenc" in output or "h264_nvenc " in output)
    return {
        "available": bool(available),
        "encoder": "h264_nvenc",
        "error": None if available else "h264_nvenc encoder was not listed by ffmpeg",
    }


def get_gpu_status() -> Dict[str, Any]:
    torch_payload: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_version": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "error": None,
    }
    try:
        import torch

        device_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        torch_payload.update(
            {
                "installed": True,
                "version": str(getattr(torch, "__version__", "")),
                "cuda_version": str(getattr(torch.version, "cuda", "") or ""),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": device_count,
                "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
            }
        )
    except Exception as exc:
        torch_payload["error"] = str(exc)

    nvidia_payload = _run_nvidia_smi()
    nvenc_payload = _check_ffmpeg_nvenc()
    ready = bool(torch_payload["cuda_available"]) and bool(nvidia_payload.get("available"))
    rendering_ready = bool(nvidia_payload.get("available")) and bool(nvenc_payload.get("available"))
    if ready and rendering_ready:
        recommendation = "GPU analysis and NVENC clip rendering are ready."
    elif ready:
        recommendation = "GPU analysis is ready. Install an FFmpeg build with h264_nvenc for GPU clip rendering."
    else:
        recommendation = "Install CUDA-enabled PyTorch and confirm nvidia-smi works."
    return {
        "ready": ready,
        "rendering_ready": rendering_ready,
        "torch": torch_payload,
        "nvidia_smi": nvidia_payload,
        "ffmpeg_nvenc": nvenc_payload,
        "recommendation": recommendation,
    }
