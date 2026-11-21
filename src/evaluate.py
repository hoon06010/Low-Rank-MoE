"""Token-weighted next-token evaluation; router auxiliary loss is excluded."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.train import _target_count, _tensor_batch


@torch.no_grad()
def evaluate(model: nn.Module, batches: Iterable[dict], *, device: str = "cpu") -> dict[str, Any]:
    selected = torch.device(device)
    if selected.type not in {'cpu', 'cuda'}:
        raise ValueError('supported evaluation devices are cpu and cuda')
    if selected.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this PyTorch environment')
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    target_tokens = 0
    try:
        for batch in batches:
            count = _target_count(batch)
            if count == 0:
                continue
            input_ids, labels, attention_mask = _tensor_batch(batch, selected)
            logits, _, _ = model(input_ids, attention_mask)
            ce_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                ignore_index=-100, reduction="sum",
            )
            if not torch.isfinite(ce_sum).all():
                raise FloatingPointError("non-finite evaluation cross-entropy")
            loss_sum += float(ce_sum.item())
            target_tokens += count
    finally:
        model.train(was_training)
    if target_tokens == 0:
        raise ValueError("evaluation data contains no valid targets")
    return {
        "loss_sum": loss_sum,
        "eval_target_tokens": target_tokens,
        "loss_nats": loss_sum / target_tokens,
    }
