import os
from pathlib import Path

import s3fs
from datetime import datetime


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def convert_model_ckpt(base_dir: Path):
    # find all ckpt files in base_dir
    ckpts = list(base_dir.rglob("*.ckpt"))
    for ckpt in ckpts:
        pass
    raise NotImplementedError


if __name__ == "__main__":
    # upload results to s3
    s3 = s3fs.S3FileSystem(
        key=os.getenv("S3_ACCESS_KEY"),
        secret=os.getenv("S3_SECRET_KEY"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    )

    s3.upload(
        "lightning_logs/",
        f"{os.getenv('S3_BUCKET')}/lightning_logs/{timestamp()}",
        recursive=True,
    )
