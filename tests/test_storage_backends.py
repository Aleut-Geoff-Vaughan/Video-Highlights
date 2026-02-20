from __future__ import annotations

from io import BytesIO
from pathlib import Path

from backend.services.storage import LocalStorageBackend, S3StorageBackend


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes) -> None:  # noqa: N803 (boto-style names)
        self.objects[(Bucket, Key)] = Body

    def generate_presigned_url(self, ClientMethod: str, Params: dict, ExpiresIn: int):  # noqa: N803
        return f"https://fake-s3.local/{Params['Bucket']}/{Params['Key']}?exp={ExpiresIn}"


def test_local_storage_backend_saves_file(tmp_path: Path) -> None:
    backend = LocalStorageBackend(root=str(tmp_path))
    obj = backend.save_file(BytesIO(b"hello"), key_prefix="match_1", filename="clip.mp4")
    assert obj.backend == "local"
    assert obj.size_bytes == 5
    path = Path(obj.path)
    assert path.exists()
    assert path.read_bytes() == b"hello"


def test_s3_storage_backend_save_and_presign() -> None:
    fake_client = FakeS3Client()
    backend = S3StorageBackend(
        bucket="vh-bucket",
        endpoint_url="https://fake-s3.local",
        region="us-east-1",
        key_prefix="vh",
        s3_client=fake_client,
    )
    obj = backend.save_file(BytesIO(b"abc123"), key_prefix="match_22", filename="video.mp4")
    assert obj.backend == "s3"
    assert obj.path.startswith("s3://vh-bucket/")
    assert obj.size_bytes == 6

    url = backend.get_download_url(obj.path, expires_seconds=600)
    assert url.startswith("https://fake-s3.local/vh-bucket/")
    assert "exp=600" in url
