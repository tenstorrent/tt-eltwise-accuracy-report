# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Changed paths to ops. Under-selecting ships a kernel nobody measured, so the fixtures
are real accuracy PRs and the docs-only one must select nothing at all."""

from __future__ import annotations

import pytest

from ttnn_accuracy.ops.select import select

MANIFEST = {
    "ops": {
        f"ttnn.{name}": {"name": name, "category": category, "operands": operands}
        for name, category, operands in (
            ("erfinv", "unary", 1),
            ("erfinv_bw", "unary_bw", 1),
            ("logaddexp", "binary", 2),
            ("logaddexp_bw", "binary_bw", 2),
            ("sqrt", "unary", 1),
            ("exp", "unary", 1),
            ("isinf", "unary", 1),
            ("isnan", "unary", 1),
            ("softcap", "unary", 1),
        )
    },
    "layouts": {
        "wh": {
            f"ttnn.{n}": {}
            for n in (
                "erfinv",
                "erfinv_bw",
                "logaddexp",
                "logaddexp_bw",
                "sqrt",
                "exp",
                "isinf",
                "isnan",
            )
        },
        "bh": {
            f"ttnn.{n}": {}
            for n in ("erfinv", "erfinv_bw", "logaddexp", "logaddexp_bw", "sqrt", "exp", "softcap")
        },
    },
}

WH_SFPU = "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu"
BH_SFPU = "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu"


def test_a_docs_only_change_selects_nothing():
    """The cheapest guard against a runaway selector, and the commonest PR shape."""
    picked = select(["README.md", "docs/tech_reports/x.md", "CODEOWNERS"], MANIFEST)
    assert picked.ops == []
    assert picked.categories == []
    assert picked.arches == []


def test_one_kernel_selects_its_op_and_its_backward_sibling():
    picked = select([f"{WH_SFPU}/ckernel_sfpu_erfinv.h"], MANIFEST)
    assert picked.ops == ["erfinv", "erfinv_bw"]
    assert picked.arches == ["wh"]
    assert picked.reasons["erfinv"].endswith("ckernel_sfpu_erfinv.h")


def test_a_kernel_on_both_architectures_selects_both():
    """erfinv #55283 is a BF16 kernel touched on Blackhole and Wormhole together."""
    picked = select(
        [f"{WH_SFPU}/ckernel_sfpu_erfinv.h", f"{BH_SFPU}/ckernel_sfpu_erfinv.h"], MANIFEST
    )
    assert picked.ops == ["erfinv", "erfinv_bw"]
    assert picked.arches == ["bh", "wh"]


def test_a_shared_header_selects_the_whole_architecture():
    """logaddexp #52856 touched ckernel_defs.h. Naming no op, it reaches every kernel — the
    blind spot that let a change ship unmeasured."""
    picked = select(["tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h"], MANIFEST)
    assert picked.ops == []
    assert "unary" in picked.categories and "binary" in picked.categories
    assert picked.arches == ["wh"]


def test_a_ttnn_family_selects_its_category_not_one_op():
    picked = select(["ttnn/cpp/ttnn/operations/eltwise/binary_ng/binary_ng.cpp"], MANIFEST)
    assert picked.categories == ["binary"]
    assert picked.ops == []


def test_quasar_is_recognised_and_skipped():
    """Real architecture, no runner here. It must not silently select wh or bh."""
    picked = select(
        ["tt_metal/hw/ckernels/quasar/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h"], MANIFEST
    )
    assert picked.arches == []
    assert picked.ops == []


def test_an_alias_header_selects_every_op_it_implements():
    picked = select([f"{WH_SFPU}/ckernel_sfpu_isinf_isnan.h"], MANIFEST)
    assert picked.ops == ["isinf", "isnan"]


def _patch(*lines: str) -> str:
    return "--- a/x\n+++ b/x\n@@ -1 +1 @@\n" + "\n".join(lines)


def test_an_alias_header_narrows_to_the_ops_its_diff_names():
    """tt-metal#59238 touched only softsign_tile in activations.h and measured nine ops."""
    path = f"{WH_SFPU}/ckernel_sfpu_isinf_isnan.h"
    hunks = {path: _patch("-    isinf_old(x);", "+    isinf_new(x);")}
    assert select([path], MANIFEST, hunks=hunks).ops == ["isinf"]


def test_an_alias_header_keeps_every_op_when_its_diff_names_none():
    """A shared macro moves them all and mentions no op, so naming none narrows nothing."""
    path = f"{WH_SFPU}/ckernel_sfpu_isinf_isnan.h"
    hunks = {path: _patch("-#define CALL(a) old(a)", "+#define CALL(a) new(a)")}
    assert select([path], MANIFEST, hunks=hunks).ops == ["isinf", "isnan"]


def test_a_diff_naming_an_op_the_alias_predates_still_narrows():
    """activations.h declares softsign, which its alias listed for years without it."""
    path = f"{WH_SFPU}/ckernel_sfpu_isinf_isnan.h"
    hunks = {path: _patch("+    calculate_exp(x);")}
    assert select([path], MANIFEST, hunks=hunks).ops == ["exp"]


def test_a_longer_op_name_does_not_name_the_one_inside_it():
    """`celu` must not read as `elu`; only `_` may sit against a name, never a letter."""
    from ttnn_accuracy.ops.select import _named_in

    assert _named_in(_patch("+    celu_tile(x);"), ("celu", "elu")) == ("celu",)
    assert _named_in(_patch("+    calculate_softsign(x);"), ("softsign",)) == ("softsign",)
    assert _named_in(_patch(" context mentions exp"), ("exp",)) == ()


def test_an_op_the_architecture_does_not_support_is_dropped():
    """softcap asserts BLACKHOLE, so a Wormhole kernel change must not dispatch it."""
    picked = select([f"{WH_SFPU}/ckernel_sfpu_softcap.h"], MANIFEST)
    assert picked.ops == []


def test_past_the_cap_it_asks_for_categories_instead(monkeypatch):
    from ttnn_accuracy.ops import select as module

    monkeypatch.setattr(module, "SELECT_CAP", 1)
    picked = module.select([f"{WH_SFPU}/ckernel_sfpu_erfinv.h"], MANIFEST)
    assert picked.capped
    assert picked.ops == []
    assert picked.categories == ["unary", "unary_bw"]


@pytest.mark.parametrize("path", ["tt_metal/hw/inc/api/compute/eltwise_unary/sqrt.h"])
def test_the_compute_api_selects_both_architectures(path):
    picked = select([path], MANIFEST)
    assert picked.ops == ["sqrt"]
    assert picked.arches == ["bh", "wh"]


def test_a_helper_header_resolves_through_whoever_includes_it(tmp_path):
    """tt_poly_log_square_factorized_odd.h names no op; erfinv_bf16.h includes it. Without
    the tree this escalates instead, which is the safe direction but a far wider run."""
    kernels = tmp_path / "tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu"
    kernels.mkdir(parents=True)
    (kernels / "ckernel_sfpu_helper.h").write_text("// no op of its own\n")
    (kernels / "ckernel_sfpu_erfinv_bf16.h").write_text('#include "ckernel_sfpu_helper.h"\n')

    changed = [f"{WH_SFPU}/ckernel_sfpu_helper.h"]
    assert select(changed, MANIFEST).unresolved == changed  # no tree: escalates
    picked = select(changed, MANIFEST, tree=tmp_path)
    assert picked.ops == ["erfinv", "erfinv_bw"]  # tree: resolved precisely
    assert picked.unresolved == []


def test_named_ops_survive_a_shared_header_in_the_same_change():
    """logaddexp #52856 touches ckernel_defs.h beside four kernels. The kernels are still
    the narrowest thing that covers it; the shared header only widens `categories`."""
    picked = select(
        [
            f"{WH_SFPU}/ckernel_sfpu_logaddexp.h",
            "tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_defs.h",
        ],
        MANIFEST,
    )
    assert picked.ops == ["logaddexp", "logaddexp_bw"]
    assert "binary" in picked.categories
