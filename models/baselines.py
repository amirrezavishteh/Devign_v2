"""Baseline methods compared in Table 2 (Sec 3.2).

Sequence baselines treat source code as a flat token sequence with word2vec-initialized
embeddings (3-layer BiLSTM [13], BiLSTM+Att, CNN [14]). They consume a token-id tensor, NOT the
graph batch, so they have their own collate path (see SequenceDataset below).

Metrics + XGBoost [25] is implemented separately in models/metrics_xgboost.py since it is a
non-neural, feature-engineering pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------------------------
# Token sequence dataset (shared by BiLSTM / CNN baselines)
# ----------------------------------------------------------------------------------------------

class TokenVocab:
    def __init__(self):
        self.tok2id = {"<PAD>": 0, "<UNK>": 1}

    def add(self, tok: str) -> int:
        if tok not in self.tok2id:
            self.tok2id[tok] = len(self.tok2id)
        return self.tok2id[tok]

    def get(self, tok: str) -> int:
        return self.tok2id.get(tok, 1)

    def __len__(self):
        return len(self.tok2id)


def tokenize_c(source: str) -> list[str]:
    """Lightweight C tokenizer: reuse tree-sitter leaf tokens in source order."""
    from data.parser import flatten_ast
    nodes, _ = flatten_ast(source)
    leaves = sorted((n for n in nodes if n.is_leaf), key=lambda n: n.start_byte)
    return [n.code for n in leaves]


class SequenceSample:
    __slots__ = ("token_ids", "label", "project")

    def __init__(self, token_ids, label, project):
        self.token_ids = token_ids
        self.label = label
        self.project = project


class SequenceDataset(Dataset):
    def __init__(self, samples: list[SequenceSample], max_len: int = 500):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_seq_collate(max_len: int = 500):
    def collate(batch: list[SequenceSample]):
        B = len(batch)
        L = min(max_len, max(len(s.token_ids) for s in batch))
        ids = torch.zeros(B, L, dtype=torch.long)
        lengths = torch.zeros(B, dtype=torch.long)
        labels = torch.zeros(B, dtype=torch.float32)
        for i, s in enumerate(batch):
            t = s.token_ids[:L]
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            lengths[i] = max(1, len(t))
            labels[i] = float(s.label)
        return {"ids": ids, "lengths": lengths, "labels": labels,
                "projects": [s.project for s in batch]}
    return collate


# ----------------------------------------------------------------------------------------------
# BiLSTM (+ optional attention)
# ----------------------------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h, mask):
        # h [B, L, dim], mask [B, L]
        scores = self.v(torch.tanh(self.w(h))).squeeze(-1)   # [B, L]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        return (h * weights).sum(dim=1)                       # [B, dim]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100, hidden_dim: int = 200,
                 layers: int = 3, attention: bool = False, dropout: float = 0.2,
                 pretrained: np.ndarray | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.use_attention = attention
        if attention:
            self.attn = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, batch) -> torch.Tensor:
        ids = batch["ids"]
        mask = ids != 0
        emb = self.embedding(ids)
        out, (hn, _) = self.lstm(emb)
        if self.use_attention:
            pooled = self.attn(out, mask)
        else:
            # mean-pool over valid timesteps
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths
        logits = self.fc(self.dropout(pooled)).squeeze(-1)
        return logits


# ----------------------------------------------------------------------------------------------
# CNN (Kim-style 1-D text CNN, [14])
# ----------------------------------------------------------------------------------------------

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100, num_filters: int = 100,
                 filter_sizes=(3, 4, 5), dropout: float = 0.2,
                 pretrained: np.ndarray | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained))
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, batch) -> torch.Tensor:
        ids = batch["ids"]
        emb = self.embedding(ids).transpose(1, 2)   # [B, embed, L]
        feats = []
        for conv in self.convs:
            c = torch.relu(conv(emb))                # [B, F, L]
            feats.append(torch.max(c, dim=2).values)  # [B, F]
        cat = torch.cat(feats, dim=1)
        logits = self.fc(self.dropout(cat)).squeeze(-1)
        return logits
