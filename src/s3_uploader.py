import os
from datetime import datetime
from pathlib import Path
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def archive_cleaned_dataset(df: pd.DataFrame, original_filename: str) -> dict:
    """
    Uploads the remediated dataset to AWS S3 if credentials exist,
    or archives locally under exports/ with a timestamped audit key.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(original_filename).stem
    export_name = f"{stem}_remediated_{timestamp}.csv"
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    bucket = os.getenv("AWS_S3_BUCKET")
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION", "ap-south-1")

    if bucket and aws_key and aws_secret:
        try:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_key,
                aws_secret_access_key=aws_secret,
                region_name=region,
            )
            s3_key = f"datasentry/remediated/{export_name}"
            s3.put_object(Bucket=bucket, Key=s3_key, Body=csv_bytes, ContentType="text/csv")
            return {"destination": "AWS S3", "location": f"s3://{bucket}/{s3_key}"}
        except Exception as exc:
            return {"destination": "Local Fallback (S3 Error)", "location": str(exc)}

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    local_path = export_dir / export_name
    local_path.write_bytes(csv_bytes)
    return {"destination": "Local Audit Vault", "location": str(local_path)}