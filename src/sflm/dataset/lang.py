import torch

from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Any
import lightning as pl


class LangABC(Dataset):
    r"""A dummy dataset of a simple formal language.

    FORMAL LANGUAGE NOTE
    Given an alphabet :math:`\Sigma`, a formal language :math:`L` is defined as a subset of :math:`\Sigma^*`.

    This is a NONDETERMINISTIC dummy dataset for generating strings in a formal language L,
    where :math:`L=\{w| |w| \leq \textit{max_len}, |w|_a + |w|_b = |w|_c\}, \Sigma=\{a, b, c\}`.

    Attributes:

    Args:
        size: The dummy size of the dataset, would affect the # for each epoch.
        max_len: The maximum length of strings defined for this language.
            Note: BOS and EOS not included.
        transform: The transform function for preprocessing.
    """

    def __init__(
        self, size: int, max_len: int, transform: Optional[Callable[[str], Any]] = None
    ) -> None:
        super().__init__()
        self.size: int = size
        r"""
        The dummy size of the dataset,
        required to specify how many steps for each epoch.
        """
        self.max_len: int = max_len
        r"""
        The maximum length of strings defined for this language.
        Note: SOS and EOS not included.        
        """
        self.transform: Optional[Callable[[str], Any]] = transform
        r"""
        The transform function for preprocessing.
        """

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Any:
        # random length from [0, max_len]
        n_c = torch.randint(self.max_len // 2 + 1, ()).item()
        n_a = torch.randint(n_c + 1, ()).item()
        n_b = n_c - n_a
        idx = torch.tensor([0] * n_a + [1] * n_b + [2] * n_c)
        idx = idx[torch.randperm(2 * n_c)]
        out = "".join(map(lambda x: "abc"[x], idx))
        if self.transform is not None:
            out = self.transform(out)
        return out


class LangABCDataModule(pl.LightningDataModule):
    r"""A LightningDataModule for LangABC dataset.

    Args:
        batch_size: The batch size.
        max_len: The maximum length of strings defined for this language.
            Note: BOS and EOS not included.
        transform: The transform function for preprocessing.
        num_workers: The number of workers for data loading.
    """

    def __init__(
        self,
        size: int,
        max_len: int,
        batch_size: int = 32,
        transform: Optional[Callable[[str], Any]] = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.size: int = size
        self.batch_size: int = batch_size
        self.max_len: int = max_len
        self.transform: Optional[Callable[[str], Any]] = transform
        self.num_workers: int = num_workers
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_set = LangABC(
                size=int(self.size * 0.8),
                max_len=self.max_len,
                transform=self.transform,
            )
            self.val_set = LangABC(
                size=int(self.size * 0.1),
                max_len=self.max_len,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_set = LangABC(
                size=int(self.size * 0.1),
                max_len=self.max_len,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
