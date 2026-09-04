"""Shape invariance, device-free: the dispatch needs silicon, the comparison does not."""

from __future__ import annotations

import torch

from ttnn_accuracy.measure import sweeps


def test_the_ragged_shape_is_actually_ragged():
    """Both dims a multiple of 32 would tile evenly and test nothing."""
    assert all(dim % 32 for dim in sweeps.RAGGED)


def test_a_result_that_moved_with_the_tiling_is_counted(monkeypatch):
    """Every sweep is 64 tiles on a 64-core grid, so an uneven split is never otherwise run."""
    shapes = []

    def fake(fn, *operands, dtype, layout, device):
        shapes.append(tuple(operands[0].shape))
        out = torch.arange(operands[0].numel(), dtype=torch.float32)
        if len(shapes) == 1:  # the uneven tiling disagrees at one element
            out[5] += 1.0
        return out

    monkeypatch.setattr(sweeps, "_on_device", fake)
    assert sweeps.shape_invariance(None, (-1.0, 1.0), 1, "bf16", "tile", None) == 1
    assert shapes[0] == sweeps.RAGGED
    assert shapes[1][1] == sweeps.TILE_WIDTH


def test_the_same_answer_in_both_tilings_is_no_finding(monkeypatch):
    monkeypatch.setattr(
        sweeps,
        "_on_device",
        lambda fn, *ops, dtype, layout, device: torch.arange(ops[0].numel(), dtype=torch.float32),
    )
    assert sweeps.shape_invariance(None, (-1.0, 1.0), 1, "bf16", "tile", None) == 0
