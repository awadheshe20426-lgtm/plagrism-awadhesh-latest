import boto3
from botocore.client import Config
import os

class StorageService:
    def __init__(self, storage_type="local"):
        self.storage_type = storage_type
        if self.storage_type == "s3":
            self.s3 = boto3.client(
                "s3",
                endpoint_url=os.getenv("S3_ENDPOINT_URL"),
                aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
                aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
                config=Config(signature_version="s3v4"),
            )
            self.bucket_name = os.getenv("S3_BUCKET_NAME")
        else:
            self.upload_dir = "uploads"
            os.makedirs(self.upload_dir, exist_ok=True)

    def save(self, filename, content):
        if self.storage_type == "s3":
            self.s3.put_object(Bucket=self.bucket_name, Key=filename, Body=content)
            return f"s3://{self.bucket_name}/{filename}"
        else:
            # Defense in depth: never allow the resolved path to escape the upload dir
            root = os.path.abspath(self.upload_dir)
            path = os.path.abspath(os.path.join(root, filename))
            if not path.startswith(root + os.sep):
                raise ValueError(f"Unsafe storage path rejected: {filename!r}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(content)
            return path

    def delete(self, filename):
        """Delete a stored object by key/path. Best-effort: a missing object is a no-op."""
        if not filename:
            return
        if self.storage_type == "s3":
            try:
                self.s3.delete_object(Bucket=self.bucket_name, Key=filename)
            except Exception:
                # Object may already be gone; deletion must be idempotent.
                pass
        else:
            root = os.path.abspath(self.upload_dir)
            path = os.path.abspath(os.path.join(root, filename))
            # Never remove anything outside the upload dir.
            if not path.startswith(root + os.sep):
                return
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    def get_presigned_url(self, filename):
        if self.storage_type == "s3":
            return self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": filename},
                ExpiresIn=3600,
            )
        else:
            # For local storage, return a relative URL that the frontend can use
            # Assuming the backend serves the 'uploads' directory at /uploads
            return f"/api/v1/files/{filename}"
