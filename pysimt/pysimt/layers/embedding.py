"""Embedding layer variants."""

from typing import Optional

import torch
from torch import nn
from . import FF


class TFEmbedding(torch.nn.Embedding):
    """Position-aware embeddings for Transformer models. Based on the original
    Transformers paper and the implementation of OpenNMT.

    Args:
        num_embeddings: The size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        max_len: Maximum known sequence length for positional encodings
        dropout: The dropout probability

    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 max_len: int = 1024, dropout: float = 0.1):
        """"""
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = dropout

        # pos_embs: (max_len, emb_dim)
        pos_embs = torch.zeros(self.max_len, self.embedding_dim)
        # pos: (max_len, 1)
        pos = torch.arange(self.max_len).unsqueeze(1)
        # divs:
        divs = torch.pow(
            10000,
            torch.arange(self.embedding_dim).float().div(self.embedding_dim))

        pos_embs[:, 0::2] = torch.sin(pos / divs[0::2])
        pos_embs[:, 1::2] = torch.cos(pos / divs[1::2])
        # pos_embs: (max_len, 1, emb_dim)
        pos_embs.unsqueeze_(1)
        sqrt_dim = torch.scalar_tensor(self.embedding_dim).sqrt()

        # Call parent's init() first
        super().__init__(num_embeddings, embedding_dim, padding_idx=0)

        # Register non-learnable params as buffers
        self.register_buffer('pos_embs', pos_embs)
        self.register_buffer('sqrt_dim', sqrt_dim)
        # Create dropout layer
        self.dropout_layer = torch.nn.Dropout(p=self.dropout)

    def forward(self, x):
        # Get the embeddings from parent's forward first
        embs = super().forward(x)
        return self.dropout_layer(
            embs.mul(self.sqrt_dim) + self.pos_embs[:embs.size(0)])
    
    def load_pretrained_vectors(self, emb_file, fixed=False):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.weight.data.copy_(pretrained)
            if fixed:
                self.weight.requires_grad = False


class ProjectedEmbedding(nn.Embedding):
    """An extension layer to regular `torch.nn.Embedding` with MLP and dropout
    applied afterwards.

    Args:
        num_embeddings: The size of the dictionary of embeddings
        embedding_dim: The size of each embedding vector
        out_dim: The output size of the feed-forward projection layer
        activ: The activation type of the feed-forward projection layer.
            `None` and `linear` denote a linear layer.
        dropout: the dropout probability

    """
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 out_dim: int, activ: Optional[str] = 'linear',
                 dropout: float = 0.0):
        """"""
        super().__init__(num_embeddings, embedding_dim, padding_idx=0)
        self.proj = FF(embedding_dim, out_dim, activ=activ, bias=False)
        self.do = nn.Dropout(dropout) if dropout > 0.0 else lambda x: x

    def forward(self, input):
        return self.do(self.proj(super().forward(input)))
