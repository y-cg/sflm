from typing import Optional
import torch
import torch.nn as nn
from torch import FloatTensor, BoolTensor


class SelfAttention(nn.Module):
    r"""A self-attention layer.

    This module takes inputs :math:`X\in\mathbb R^{N\times L\times D_e}`, and projects them into
    queries :math:`Q\in\mathbb R^{N\times L\times D_k}`, keys :math:`K\in\mathbb R^{N\times L\times D_k}`,
    and values :math:`V\in\mathbb R^{N\times L\times D_v}`, where
    :math:`N` is the batch size, :math:`L` is the padded sequence length.
    Accordingly, the layer outputs in shape :math:`(N, L, D_v)`.

    Args:
        emb_dim: The dimension of embeddings, i.e. D_e.
        key_dim: The dimension of keys, i.e. D_k.
        val_dim: The dimension of values, i.e. D_v.
    """

    def __init__(self, emb_dim: int, key_dim: int, val_dim: int) -> None:
        super().__init__()

        self.emb_dim: int = emb_dim
        r"""
        The dimension of embedings, i.e. :math:`D_e`.
        """
        self.key_dim: int = key_dim
        r"""
        The dimension of keys, i.e. :math:`D_k`.
        r"""
        self.val_dim: int = val_dim
        r"""
        The dimension of values, i.e. :math:`D_v`.
        """

        self.proj_q: nn.Parameter = nn.Parameter(
            torch.empty(self.emb_dim, self.key_dim)
        )
        r"""
        The projection matrix for queries :math:`M_q\in\mathbb R^{D_e, D_v}`.
        """
        self.proj_k: nn.Parameter = nn.Parameter(
            torch.empty(self.emb_dim, self.key_dim)
        )
        r"""
        The projection matrix for keys :math:`M_k\in\mathbb R^{D_e, D_k}`.
        """
        self.proj_v: nn.Parameter = nn.Parameter(
            torch.empty(self.emb_dim, self.val_dim)
        )
        r"""
        The projection matrix for values :math:`M_v\in\mathbb R^{D_e, D_v}`.
        """

        nn.init.xavier_uniform_(self.proj_q)
        nn.init.xavier_uniform_(self.proj_k)
        nn.init.xavier_uniform_(self.proj_v)

    def forward(
        self, x: FloatTensor, attn_mask: Optional[BoolTensor] = None
    ) -> FloatTensor:
        r"""Compute self-attention output.

        Args:
            x: The input of shape :math:`(N, L, D_e)`.
            attn_mask: The optional attention mask of shape :math:`(N, L, L)`.

        Returns:
            The self-attention output of shape :math:`(N, L, D_v)`.
        """
        # Validate arguments.
        assert x.dim() == 3 and x.size(-1) == self.emb_dim, (
            "The input shape should be of (N, L, D_e)!"
        )

        N, L, D_e = x.size()
        if attn_mask is not None:
            assert attn_mask.size() == (N, L, L), (
                "The attention mask must be of shape (N, L, L)!"
            )
            assert attn_mask.dtype == torch.bool, (
                "The attention mask must be a boolean tensor!"
            )
        else:
            attn_mask = torch.ones(N, L, L, dtype=torch.bool, device=x.device)

        # Compute self-attention output.
        Q = x @ self.proj_q  # (N, L, D_e) @ (D_e, D_k) -> (N, L, D_k)
        K = x @ self.proj_k  # (N, L, D_e) @ (D_e, D_k) -> (N, L, D_k)
        V = x @ self.proj_v  # (N, L, D_e) @ (D_e, D_v) -> (N, L, D_v)
        # score = Q @ K^T / sqrt(d_model)
        score = (
            Q @ K.transpose(-2, -1) / (D_e**0.5)
        )  # (N, L, D_k) @ (N, D_k, L) -> (N, L, L)
        # mask the positions where attn_mask is false
        score = score.masked_fill(~attn_mask, float("-inf"))  # (N, L, L)
        # weights = softmax(score)
        attn_weights = torch.functional.F.softmax(score, -1)  # (N, L, L)
        # out = weights @ V
        out = attn_weights @ V  # (N, L, L) @ (N, L, D_v) -> (N, L, D_v)
        return out
