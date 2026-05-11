from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "X-Request-ID" in response.headers


def test_gpu_health_shape(client: TestClient) -> None:
    response = client.get("/v1/health/gpu")

    assert response.status_code == 200
    payload = response.json()
    assert "ready" in payload
    assert "rendering_ready" in payload
    assert "torch" in payload
    assert "nvidia_smi" in payload
    assert "ffmpeg_nvenc" in payload
    assert "recommendation" in payload
