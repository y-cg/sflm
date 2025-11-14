import os
from pathlib import Path

import s3fs
from datetime import datetime

import torch
from dotenv import load_dotenv

from sflm.model.sflm import SFLM


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d%H%M")


def convert_model_ckpt(base_dir: Path):
    # find all ckpt files in base_dir
    ckpts = list(base_dir.rglob("*.ckpt"))
    for ckpt in ckpts:
        model = SFLM.load_from_checkpoint(ckpt)
        torch.save(
            dict(
                model_config={
                    "vocab_size": model.vocab_size,
                    "emb_dim": model.emb_dim,
                    "block_size": model.block_size,
                },
                state_dict=model.state_dict(),
            ),
            f"{ckpt.parent}/model.sav",
        )
        print(f"Converted {ckpt} to {ckpt.parent}/model.sav")


if __name__ == "__main__":
    # dotenv injection
    load_dotenv()

    convert_model_ckpt(Path("lightning_logs/"))
    # upload results to s3
    s3 = s3fs.S3FileSystem(
        key=os.getenv("S3_ACCESS_KEY"),
        secret=os.getenv("S3_SECRET_KEY"),
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    )

    s3.upload(
        "lightning_logs/",
        f"{os.getenv('S3_BUCKET_NAME')}/lightning_logs-{timestamp()}",
        recursive=True,
    )
