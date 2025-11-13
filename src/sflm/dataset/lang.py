import torch

from torch.utils.data import Dataset
from typing import Optional, Callable, Any


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
