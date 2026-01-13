"""
Cloudflare R2 Client for uploading videos.

Required environment variables:
- R2_ENDPOINT: https://ACCOUNT_ID.r2.cloudflarestorage.com
- R2_ACCESS_KEY_ID: Your R2 access key
- R2_SECRET_ACCESS_KEY: Your R2 secret key
- R2_BUCKET_NAME: Your bucket name
- R2_PUBLIC_URL: (Optional) Public URL for the bucket
"""

import boto3
from botocore.config import Config
import os


class R2Client:
    def __init__(self):
        """Initialize the R2 client using environment variables."""

        # Validate required environment variables
        required_vars = ["R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Create S3-compatible client for R2
        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4"),
            region_name="auto"
        )

        self.bucket = os.environ["R2_BUCKET_NAME"]
        self.public_url = os.environ.get("R2_PUBLIC_URL", "")

        print(f"R2 Client initialized for bucket: {self.bucket}")

    def upload_video(self, video_bytes: bytes, filename: str) -> str:
        """
        Upload video bytes to R2 and return the public URL.

        Args:
            video_bytes: The video file content as bytes
            filename: The filename to use (e.g., "uuid.mp4")

        Returns:
            The public URL of the uploaded video
        """
        key = f"videos/{filename}"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=video_bytes,
            ContentType="video/mp4"
        )

        # Return public URL if configured, otherwise construct S3-style URL
        if self.public_url:
            url = f"{self.public_url.rstrip('/')}/{key}"
        else:
            # Fallback to R2 public URL format (if bucket is public)
            url = f"https://{self.bucket}.r2.dev/{key}"

        print(f"Uploaded video to: {url}")
        return url

    def generate_presigned_url(self, filename: str, expires_in: int = 3600) -> str:
        """
        Generate a presigned URL for downloading a video.

        Args:
            filename: The filename (without 'videos/' prefix)
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            A presigned URL that allows temporary access to the file
        """
        key = f"videos/{filename}"

        url = self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in
        )

        return url
