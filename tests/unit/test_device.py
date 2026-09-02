"""Device refusals are published verbatim, so they must not carry whoever built tt-metal."""

from __future__ import annotations

from ttnn_accuracy.measure.device import reason

FATAL = (
    "TT_FATAL @ /localdev/ijankowski/tt-metal/ttnn/cpp/ttnn/operations/eltwise/unary/"
    "device/unary_device_operation.cpp:71: input_tensor.device()->arch() == tt::ARCH::BLACKHOLE"
)


def test_the_builders_home_directory_is_stripped(monkeypatch):
    monkeypatch.delenv("TT_METAL_HOME", raising=False)
    out = reason(Exception(FATAL))
    assert "ijankowski" not in out and "/localdev" not in out
    assert out.startswith("TT_FATAL @ ttnn/cpp/ttnn/operations/eltwise")
    assert "unary_device_operation.cpp:71" in out  # the useful half survives


def test_tt_metal_home_is_stripped_when_it_differs_from_the_build_path(monkeypatch):
    monkeypatch.setenv("TT_METAL_HOME", "/opt/tt-metal")
    assert reason(Exception("failed at /opt/tt-metal/build/lib.so")) == "failed at build/lib.so"


def test_only_the_first_line_is_kept():
    assert reason(Exception("first\nsecond")) == "first"
    assert reason(Exception("")) == ""
