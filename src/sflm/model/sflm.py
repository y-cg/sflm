import torch
import torch.nn as nn

from torch import FloatTensor, LongTensor, BoolTensor
from typing import Optional


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


class SFLM(nn.Module):
    r"""A Small formal language model.

    This module takes sequences of indices of tokens in shape :math:`(N, L)`, where :math:`N` is the batch size,
    :math:`L` is the padded length of sequences, and outputs logits of shape :math:`(N, L, V)` for predicting next
    token at each position, where :math:`L` is the size of the vocabulary.

    Args:
        vocab_size: The size of the vocabulary, i.e. :math:`V`.
        emb_dim: The dimension of embeddings.
        block_size: The maximum length of input sequences, i.e. maximum value of :math:`L`.
    """

    def __init__(self, vocab_size: int, emb_dim: int, block_size: int) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        r"""The size of the vocabulary, i.e. V."""
        self.emb_dim: int = emb_dim
        r"""The dimension of embeddings."""
        self.block_size: int = block_size
        r"""The maximum length of input sequences."""

        self.tok_embedding: nn.Embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.emb_dim
        )
        r"""The embedding layer for translating token indices into embeddings."""
        self.pos_embedding: nn.Embedding = nn.Embedding(
            num_embeddings=self.block_size, embedding_dim=self.emb_dim
        )
        r"""The embedding layer for encoding the position of the corresponding token."""
        self.self_attention: SelfAttention = SelfAttention(
            emb_dim=self.emb_dim, key_dim=self.emb_dim, val_dim=self.emb_dim
        )
        r"""The self-attention layer. Here, the dimension of keys and values are set to emb_dim."""
        self.layer_norm_1: nn.LayerNorm = nn.LayerNorm(self.emb_dim)
        r"""The layer normalization for self-attention layer."""
        self.fnn: nn.Sequential = nn.Sequential(
            nn.Linear(self.emb_dim, 4 * self.emb_dim),
            nn.ReLU(),
            nn.Linear(4 * self.emb_dim, self.emb_dim),
        )
        r"""The FNN layer."""
        self.layer_norm_2: nn.LayerNorm = nn.LayerNorm(self.emb_dim)
        r"""The layer normalization for FNN layer."""
        self.head: nn.Linear = nn.Linear(self.emb_dim, self.vocab_size)
        r"""The linear layer project embeddings into logits for predicting the next token at each position."""

    def forward(self, idx: LongTensor) -> FloatTensor:
        r"""Compute logits of the next token.

        Denote input as :math:`S`, and output as :math:`Z`,
        :math:`Z_{i,j,k}` represents the logit of :math:`S_{i,j+1}`
        to be the :math:`k`-th token given :math:`S_{i,1:j}`.

        The model must be **CAUSAL** in the aspect of the sequence order
        i.e. at each position, the model cannot access any token after it.
        Achieve this requirement by applying a proper attention mask.

        Args:
            idx: The input of shape :math:`(N, L)`.

        Returns:
            The language model output of shape :math:`(N, L, V)` for predicting the next token.
        """
        # Validate arguments.
        assert idx.dim() == 2, "Indices must be of shape (N, L)!"
        assert idx.dtype == torch.long, "Indices must be of type Long!"
        N, L = idx.size()
        assert L <= self.block_size, "Sequences are too long!"

        # Compute the SFLM output.
        # word -> embedding
        tok_emb = self.tok_embedding(idx)
        # position -> pos embedding
        pos_emb = self.pos_embedding(idx)
        # synthesize embeddings
        x = tok_emb + pos_emb
        # causal attention mask
        # [1, 0, 0, ...]
        # [1, 1, 0, ...]
        # [1, 1, 1, ...]
        causal_mask = torch.tril(torch.ones(L, L)).bool()
        # expand to (N, L, L)
        causal_mask = causal_mask.unsqueeze(0).expand(N, -1, -1)
        # self-attention
        attn = self.self_attention(x, causal_mask)
        # residual connection + layer norm
        x = self.layer_norm_1(x + attn)
        # FNN
        fnn = self.fnn(x)
        # residual connection + layer norm
        x = self.layer_norm_2(x + fnn)
        # project to logits
        logit = self.head(x)
        return logit

    @torch.no_grad()
    def generate(
        self, cond_idx: LongTensor, steps: int, temperature: Optional[float] = 1.0
    ) -> LongTensor:
        r"""Conditional sample from this language model.

        Given a single BOS as the condition, a sample "abcc" is generated.

        Args:
            cond_idx: The input of shape :math:`(N, L)`.
                It represents the indices of the first :math:`L` token given as condition.
            steps: The steps for generation.
            temperature: The temperature for sampling, default to 1.0.
                For greedy strategy, just give 0.0.

        Returns:
            The sampled indices of shape :math:`(N, L + \textit{steps})`. When :math:`L + \textit{steps}` is greater than
            block_size, the generation would always depends on the last block_size tokens in a moving window.
        """
        assert cond_idx.dim() == 2, "Condition indices must be of shape (N, L)!"
        assert cond_idx.dtype == torch.long, "Condition indices must be of type Long!"
        assert temperature >= 0, "Temperature cannot be less than zero by definition!"
        N, L = cond_idx.size()

        idx = cond_idx.clone()
        for _ in range(steps):
            logit = self(idx[:, -self.block_size :])
            next_token_logit = logit[:, -1, :]
            if temperature > 0:
                next_token_prob = torch.softmax(next_token_logit / temperature, -1)
                next_token_id = torch.multinomial(next_token_prob, num_samples=1)
            else:
                next_token_id = torch.argmax(next_token_logit, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_token_id], -1)
        return idx
