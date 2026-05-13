from __future__ import annotations

import subprocess

import numpy as np

from backend.services.media_timeline import generate_waveform_peaks


class _FakeCompletedProcess:
    def __init__(self, stdout: bytes, returncode: int = 0, stderr: bytes = b"") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def test_generate_waveform_peaks_normalizes_audio_bins(monkeypatch) -> None:
    samples = np.asarray([0.0, 0.2, -0.5, 1.0, -0.25, 0.1, 0.0, -0.75], dtype=np.float32)

    def _fake_run(command, capture_output, check, timeout):  # noqa: ANN001
        assert "-f" in command
        assert "f32le" in command
        assert command[-1] == "pipe:1"
        assert capture_output is True
        assert check is False
        assert timeout >= 30.0
        return _FakeCompletedProcess(samples.tobytes())

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = generate_waveform_peaks("C:/tmp/source.mp4", bins=16, duration_seconds=4.0)

    assert result["error"] is None
    assert result["peaks"] == [0.0, 0.2, 0.5, 1.0, 0.25, 0.1, 0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_generate_waveform_peaks_returns_error_when_ffmpeg_fails(monkeypatch) -> None:
    def _fake_run(command, capture_output, check, timeout):  # noqa: ANN001
        return _FakeCompletedProcess(b"", returncode=1, stderr=b"no audio stream")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = generate_waveform_peaks("C:/tmp/source.mp4", bins=4)

    assert result["peaks"] == []
    assert result["error"] == "no audio stream"
