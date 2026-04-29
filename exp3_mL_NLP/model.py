"""TextCNN model for IMDB sentiment analysis.

Reference: Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """TextCNN for binary sentiment classification.

    Architecture:
        Embedding -> Parallel Conv1d (kernel=3,4,5) -> GlobalMaxPool -> Concat -> Dropout -> FC -> Output

    Args:
        vocab_size: Size of vocabulary.
        embed_dim: Dimension of word embeddings.
        num_filters: Number of filters per conv kernel size.
        filter_sizes: List of kernel sizes for Conv1d layers.
        num_classes: Number of output classes (2 for binary).
        dropout: Dropout probability.
        pad_idx: Index of padding token.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        num_filters: int = 100,
        filter_sizes: tuple = (3, 4, 5),
        num_classes: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with uniform distribution."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of token indices, shape (batch_size, seq_len).

        Returns:
            Logits tensor, shape (batch_size, num_classes).
        """
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        emb = self.embedding(x)
        # (batch, seq_len, embed_dim) -> (batch, embed_dim, seq_len)
        emb = emb.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            # (batch, embed_dim, seq_len) -> (batch, num_filters, seq_len - k + 1)
            out = conv(emb)
            out = F.relu(out)
            # Global max pooling -> (batch, num_filters, 1)
            out = F.max_pool1d(out, kernel_size=out.shape[2])
            conv_outputs.append(out.squeeze(2))  # (batch, num_filters)

        # Concatenate all filter outputs -> (batch, num_filters * num_filter_sizes)
        cat = torch.cat(conv_outputs, dim=1)
        cat = self.dropout(cat)

        return self.fc(cat)  # (batch, num_classes)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
