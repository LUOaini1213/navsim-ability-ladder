"""Controlled current-object encoders; no new architecture claim."""
from __future__ import annotations

import torch


class MechanismModel(torch.nn.Module):
    def __init__(self, mode, hidden=512, layers=3, object_hidden=32):
        super().__init__()
        if mode not in ("none", "concat", "lead", "set"):
            raise ValueError(mode)
        self.mode = mode
        dimension = 88
        if mode == "concat":
            dimension += 88
        elif mode == "lead":
            dimension += 2
        elif mode == "set":
            self.object_encoder = torch.nn.Sequential(torch.nn.Linear(10, object_hidden), torch.nn.ReLU(),
                                                       torch.nn.Linear(object_hidden, object_hidden), torch.nn.ReLU())
            dimension += 2*object_hidden
        blocks = []
        for _ in range(layers):
            blocks.extend([torch.nn.Linear(dimension, hidden), torch.nn.ReLU()])
            dimension = hidden
        blocks.append(torch.nn.Linear(dimension, 24))
        self.head = torch.nn.Sequential(*blocks)

    def forward(self, context, objects):
        parts = [context[:, :88]]
        if self.mode == "lead":
            parts.append(context[:, 88:90])
        elif self.mode == "concat":
            parts.append(objects.flatten(1))
        elif self.mode == "set":
            mask = objects[:, :, 10:11] > .5
            encoded = self.object_encoder(objects[:, :, :10])
            mean = (encoded*mask).sum(1) / mask.sum(1).clamp(min=1)
            maximum = encoded.masked_fill(~mask, -torch.inf).max(1).values
            maximum = torch.where(mask.any(1), maximum, torch.zeros_like(maximum))
            parts.extend([mean, maximum])
        return self.head(torch.cat(parts, dim=1))
