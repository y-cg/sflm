from sflm.dataset import LangABCDataModule
from sflm.model.sflm import SFLM
from sflm.utils import TokenizerABC
import lightning as pl
from lightning.pytorch import seed_everything

if __name__ == "__main__":
    seed_everything(42, workers=True)

    str_len = 10
    embed_dim = 16
    batch_size = 32

    tokenizer = TokenizerABC()

    def tokenize(x):
        return tokenizer.encode(x, pad_to_length=str_len + 2)

    dm = LangABCDataModule(
        size=10000,
        max_len=str_len,
        batch_size=batch_size,
        transform=tokenize,
    )

    model = SFLM(
        vocab_size=tokenizer.num_tokens,
        emb_dim=embed_dim,
        block_size=str_len + 1,
    )

    trainer = pl.Trainer()

    trainer.fit(model, datamodule=dm)
