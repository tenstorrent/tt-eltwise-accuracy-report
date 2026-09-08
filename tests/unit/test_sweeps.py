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


def test_only_the_sampled_sweeps_have_anything_to_refine():
    """`refine` reads the same table the pages print, so neither can claim the other's coverage."""
    assert set(sweeps.SAMPLED) == {(2, "fp32"), (3, "bf16"), (3, "fp32")}
    assert all((1, d) not in sweeps.SAMPLED for d in ("bf16", "fp32"))  # unary is exhaustive
    assert (2, "bf16") not in sweeps.SAMPLED  # every bf16 pair is measured


def test_the_candidates_are_the_cell_the_sample_drew_from():
    """fp32/pow published 5918 ULP; sweeping this cell found 8300 at the same point."""
    fp32 = sweeps._neighbours(1.5, "fp32")
    assert fp32.numel() == 2**16
    assert float(fp32.min()) == 1.5 and float(fp32.max()) < 1.508
    assert sweeps._neighbours(1.5, "bf16").numel() == 2**16  # the whole bf16 space
