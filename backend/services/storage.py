from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

from ..config import settings
from ..utils import ensure_dir, generate_id


@dataclass
class StoredObject:
    object_id: str
    path: str
    size_bytes: int
    backend: str


class StorageBackend:
    def save_file(self, stream: BinaryIO, key_prefix: str, filename: str) -> StoredObject:
        raise NotImplementedError

    def get_download_url(self, stored_path: str, expires_seconds: int = 3600) -> str:
        return stored_path


class LocalStorageBackend(StorageBackend):
    def __init__(self, root: str) -> None:
        self.root = Path(ensure_dir(root))

    def save_file(self, stream: BinaryIO, key_prefix: str, filename: str) -> StoredObject:
        obj_id = generate_id("asset")
        folder = Path(ensure_dir(str(self.root / key_prefix)))
        safe_name = filename or "upload.bin"
        out_path = folder / f"{obj_id}_{safe_name}"
        size_bytes = 0
        with out_path.open("wb") as handle:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                size_bytes += len(chunk)
        return StoredObject(
            object_id=obj_id,
            path=str(out_path.resolve()),
            size_bytes=size_bytes,
            backend="local",
        )


class S3StorageBackend(StorageBackend):
    def __init__(
        self,
        bucket: str,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        key_prefix: str = "video-highlights",
        s3_client: object | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket is required")
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.region = region
        self.key_prefix = key_prefix.strip("/") if key_prefix else ""
        if s3_client is not None:
            self.client = s3_client
            return
        try:
            import boto3
        except Exception as exc:
            raise RuntimeError("boto3 is required for S3 storage backend") from exc

        kwargs = {
            "service_name": "s3",
            "endpoint_url": endpoint_url,
            "region_name": region,
            "aws_access_key_id": access_key_id,
            "aws_secret_access_key": secret_access_key,
        }
        # Drop None values for cleaner boto3 initialization.
        kwargs = {k: v for k, v in kwargs.items() if v}
        self.client = boto3.client(**kwargs)

    def save_file(self, stream: BinaryIO, key_prefix: str, filename: str) -> StoredObject:
        obj_id = generate_id("asset")
        safe_name = filename or "upload.bin"
        combined_prefix = "/".join(part.strip("/") for part in [self.key_prefix, key_prefix] if part)
        key = f"{combined_prefix}/{obj_id}_{safe_name}" if combined_prefix else f"{obj_id}_{safe_name}"

        data = stream.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        size_bytes = len(data or b"")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data or b"")
        return StoredObject(
            object_id=obj_id,
            path=f"s3://{self.bucket}/{key}",
            size_bytes=size_bytes,
            backend="s3",
        )

    def get_download_url(self, stored_path: str, expires_seconds: int = 3600) -> str:
        prefix = f"s3://{self.bucket}/"
        if not stored_path.startswith(prefix):
            return stored_path
        key = stored_path[len(prefix) :]
        return self.client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=max(60, int(expires_seconds)),
        )


def get_storage_backend() -> StorageBackend:
    if settings.storage_backend == "s3":
        return S3StorageBackend(
            bucket=settings.s3_bucket or "",
            endpoint_url=settings.s3_endpoint_url,
            region=settings.s3_region,
            access_key_id=settings.s3_access_key_id,
            secret_access_key=settings.s3_secret_access_key,
            key_prefix=settings.s3_key_prefix,
        )
    return LocalStorageBackend(settings.local_storage_root)
