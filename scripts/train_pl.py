from argparse import ArgumentParser

from torch import Tensor

from sflm.dataset import LangABCDataModule
from sflm.model.sflm import SFLM
from sflm.utils import TokenizerABC
import lightning as pl
from lightning.pytorch import seed_everything
from dotenv import load_dotenv
import torch


def config():
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset-size",
        type=int,
        default=1_000_000,
        help="The dummy size of the dataset (as samples are generated nondeterministically).",
    )

    parser.add_argument(
        "--str-len",
        type=int,
        default=20,
        help="The length of strings sampled from the dataset.",
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=16,
        help="The dimension of embeddings applied in the model.",
    )

    parser.add_argument(
        "--epochs", type=int, default=20, help="The # of 'epochs' of training."
    )

    parser.add_argument("--lr", type=float, default=0.01, help="The learning rate.")
    parser.add_argument("--wd", type=float, default=0.0, help="The weight decay.")
    parser.add_argument("--batch-size", type=int, default=64, help="The batch size.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed.")

    return parser.parse_args()


if __name__ == "__main__":
    # dotenv injection
    load_dotenv()

    # parse config
    config = config()

    seed_everything(config.seed, workers=True)
    torch.use_deterministic_algorithms(True)

    tokenizer = TokenizerABC()

    def tokenize(x) -> Tensor:
        return tokenizer.encode(x, pad_to_length=config.str_len + 2)

    dm = LangABCDataModule(
        size=config.dataset_size,
        max_len=config.str_len,
        batch_size=config.batch_size,
        transform=tokenize,
    )

    model = SFLM(
        vocab_size=tokenizer.num_tokens,
        emb_dim=config.embed_dim,
        block_size=config.str_len + 1,
        lr=config.lr,
        weight_decay=config.wd,
    )

    trainer = pl.Trainer(
        max_epochs=config.epochs,
    )

    trainer.fit(model, datamodule=dm)
