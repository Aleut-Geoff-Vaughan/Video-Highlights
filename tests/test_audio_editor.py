from __future__ import annotations

import pytest

from backend.config import settings
from backend.services.audio_editor import build_audio_edit_command


def test_build_audio_edit_command_removes_audio() -> None:
    command = build_audio_edit_command(
        source_video_path="C:/tmp/source.mp4",
        output_path="C:/tmp/no-audio.mp4",
        mode="remove",
    )

    assert "-an" in command
    assert "-c:v" in command
    assert "copy" in command
    assert "C:/tmp/no-audio.mp4" in command


def test_build_audio_edit_command_mixes_mp3_with_cleaned_original() -> None:
    command = build_audio_edit_command(
        source_video_path="C:/tmp/source.mp4",
        output_path="C:/tmp/mixed.mp4",
        mode="mix",
        cleanup_profile="wind_reduce",
        external_audio_path="C:/tmp/music.mp3",
        original_volume=0.8,
        music_volume=0.25,
    )
    joined = " ".join(command)

    assert "-stream_loop" in command
    assert "C:/tmp/music.mp3" in command
    assert "-filter_complex" in command
    assert "highpass=f=180" in joined
    assert "amix=inputs=2" in joined
    assert "volume=0.800" in joined
    assert "volume=0.250" in joined


def test_build_audio_edit_command_requires_external_audio_for_replace() -> None:
    with pytest.raises(ValueError, match="requires an uploaded"):
        build_audio_edit_command(
            source_video_path="C:/tmp/source.mp4",
            output_path="C:/tmp/replaced.mp4",
            mode="replace",
        )


def test_build_audio_edit_command_requires_rnnoise_model(monkeypatch) -> None:
    monkeypatch.setattr(settings, "rnnoise_model_path", None)

    with pytest.raises(ValueError, match="VH_RNNOISE_MODEL_PATH"):
        build_audio_edit_command(
            source_video_path="C:/tmp/source.mp4",
            output_path="C:/tmp/ai-clean.mp4",
            mode="keep",
            cleanup_profile="ai_rnnoise",
        )
