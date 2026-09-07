#!/usr/bin/env python3
# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "h5py==3.16.0",
#     "hdf5plugin==6.0.0",
#     "numpy==2.4.4",
# ]
# ///
"""Regenerate ZFP crosscheck fixtures from the reference H5Z-ZFP plugin.

Produces a `manifest.json` plus three files per fixture:
  <name>.raw.bin          raw uncompressed values (little-endian native width)
  <name>.compressed.bin   reference compressed chunk bytes (single chunk)
  <name>.cd.bin           cd_values as u32 little-endian tuple (informational)

The pinned versions above are the exact toolchain that produced the committed
fixtures: hdf5plugin 6.0.0 bundles H5Z-ZFP 1.1.1 / ZFP 1.0.1, matching the
`filter_name` recorded in `manifest.json`. Keep them in sync if you bump either.

Run from the repo root with uv (resolves the pins above automatically):
    uv run tests/data/h5py/zfp/regen.py

Or in a venv pinned from the sibling requirements file:
    python -m venv tests/data/h5py/zfp/.venv
    tests/data/h5py/zfp/.venv/bin/pip install -r tests/data/h5py/zfp/requirements.txt
    tests/data/h5py/zfp/.venv/bin/python tests/data/h5py/zfp/regen.py
"""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent
MANIFEST = FIXTURE_DIR / "manifest.json"


@dataclass
class Case:
    name: str
    dtype: str  # "f32", "f64", "i32", "i64"
    shape: tuple[int, ...]
    data: Any  # numpy array
    rate: float
    notes: str = ""

    def to_numpy(self) -> np.ndarray:
        np_dtype = {
            "f32": np.float32,
            "f64": np.float64,
            "i32": np.int32,
            "i64": np.int64,
        }[self.dtype]
        return np.asarray(self.data, dtype=np_dtype).reshape(self.shape)


def build_cases() -> list[Case]:
    cases: list[Case] = []

    # ---- 1D ----
    # ramp, 16 and 32 elements
    for dtype in ("f32", "f64"):
        for n in (16, 32):
            cases.append(
                Case(
                    name=f"{dtype}_1d_{n}_ramp_rate16",
                    dtype=dtype,
                    shape=(n,),
                    data=np.arange(n, dtype=np.float64) * 0.25,
                    rate=16.0,
                    notes="monotonic ramp",
                )
            )
            cases.append(
                Case(
                    name=f"{dtype}_1d_{n}_ramp_rate8",
                    dtype=dtype,
                    shape=(n,),
                    data=np.arange(n, dtype=np.float64) * 0.25,
                    rate=8.0,
                )
            )
    # near-lossless rates
    cases.append(
        Case(
            name="f32_1d_16_ramp_rate32",
            dtype="f32",
            shape=(16,),
            data=np.arange(16, dtype=np.float64) * 0.25,
            rate=32.0,
            notes="max rate for f32 (near lossless)",
        )
    )
    cases.append(
        Case(
            name="f64_1d_16_ramp_rate64",
            dtype="f64",
            shape=(16,),
            data=np.arange(16, dtype=np.float64) * 0.25,
            rate=64.0,
            notes="max rate for f64 (near lossless)",
        )
    )

    # edge: all zeros
    cases.append(
        Case(
            name="f32_1d_16_zeros_rate16",
            dtype="f32",
            shape=(16,),
            data=np.zeros(16),
            rate=16.0,
        )
    )
    # edge: mixed signs
    cases.append(
        Case(
            name="f32_1d_16_mixed_rate16",
            dtype="f32",
            shape=(16,),
            data=np.array([1, -2, 3, -4] * 4, dtype=np.float64),
            rate=16.0,
        )
    )
    # edge: partial block (not a multiple of 4)
    cases.append(
        Case(
            name="f32_1d_13_partial_rate16",
            dtype="f32",
            shape=(13,),
            data=np.arange(13, dtype=np.float64),
            rate=16.0,
            notes="length not a multiple of 4 -> partial block",
        )
    )
    # edge: single partial block of mixed magnitudes at a high rate, which
    # historically exposed a reference-diff in the decoder.
    cases.append(
        Case(
            name="f32_1d_4_varied_rate24",
            dtype="f32",
            shape=(4,),
            data=np.array([100.0, -50.5, 42.0, 80.0], dtype=np.float64),
            rate=24.0,
            notes="mixed-magnitude block that exposes a reference-diff at rate 24",
        )
    )

    # ---- 1D integers ----
    for dtype in ("i32", "i64"):
        cases.append(
            Case(
                name=f"{dtype}_1d_16_ramp_rate16",
                dtype=dtype,
                shape=(16,),
                data=np.arange(16),
                rate=16.0,
            )
        )
        cases.append(
            Case(
                name=f"{dtype}_1d_16_ramp_rate32",
                dtype=dtype,
                shape=(16,),
                data=np.arange(16),
                rate=32.0,
            )
        )

    # ---- 2D ----
    for dtype in ("f32", "f64", "i32", "i64"):
        cases.append(
            Case(
                name=f"{dtype}_2d_4x4_ramp_rate16",
                dtype=dtype,
                shape=(4, 4),
                data=np.arange(16),
                rate=16.0,
            )
        )
        cases.append(
            Case(
                name=f"{dtype}_2d_8x12_ramp_rate16",
                dtype=dtype,
                shape=(8, 12),
                data=np.arange(96),
                rate=16.0,
            )
        )
    # 2D edge: shape not divisible by 4 on each axis
    cases.append(
        Case(
            name="f32_2d_5x7_partial_rate16",
            dtype="f32",
            shape=(5, 7),
            data=np.arange(35, dtype=np.float64),
            rate=16.0,
            notes="neither axis divisible by 4",
        )
    )

    # ---- 3D ----
    for dtype in ("f32", "f64", "i32", "i64"):
        cases.append(
            Case(
                name=f"{dtype}_3d_4x4x4_ramp_rate16",
                dtype=dtype,
                shape=(4, 4, 4),
                data=np.arange(64),
                rate=16.0,
            )
        )
        cases.append(
            Case(
                name=f"{dtype}_3d_8x8x8_ramp_rate16",
                dtype=dtype,
                shape=(8, 8, 8),
                data=np.arange(512),
                rate=16.0,
            )
        )

    # ---- 4D ----
    for dtype in ("f32", "f64", "i32", "i64"):
        cases.append(
            Case(
                name=f"{dtype}_4d_4x4x4x4_ramp_rate16",
                dtype=dtype,
                shape=(4, 4, 4, 4),
                data=np.arange(256),
                rate=16.0,
            )
        )

    return cases


def write_fixture(case: Case) -> dict[str, Any]:
    arr = case.to_numpy()
    h5_path = FIXTURE_DIR / f"{case.name}.h5"

    # Single chunk matching the full shape so the fixture holds exactly one
    # compressed payload we can isolate.
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "v",
            data=arr,
            chunks=arr.shape,
            **hdf5plugin.Zfp(rate=case.rate),
        )

    # Read back to extract the single chunk's raw bytes + cd_values.
    with h5py.File(h5_path, "r") as f:
        d = f["v"]
        dcpl = d.id.get_create_plist()
        filter_id, filter_flags, cd_values, filter_name = dcpl.get_filter(0)
        chunk_offset = tuple(0 for _ in arr.shape)
        filter_mask, chunk_bytes = d.id.read_direct_chunk(chunk_offset)

    # Sanity check
    assert filter_id == 32013, f"expected ZFP id, got {filter_id}"
    assert filter_mask == 0, "filter mask should be 0 (all filters applied)"

    raw_bytes = arr.tobytes(order="C")
    (FIXTURE_DIR / f"{case.name}.raw.bin").write_bytes(raw_bytes)
    (FIXTURE_DIR / f"{case.name}.compressed.bin").write_bytes(chunk_bytes)
    cd_bytes = b"".join(struct.pack("<I", int(v) & 0xFFFFFFFF) for v in cd_values)
    (FIXTURE_DIR / f"{case.name}.cd.bin").write_bytes(cd_bytes)

    # The .h5 is useful as a reference artifact for manual inspection but the
    # tests only need the three .bin files, so keep .h5 too so humans can
    # `h5dump` it.

    return {
        "name": case.name,
        "dtype": case.dtype,
        "shape": list(case.shape),
        "rate": case.rate,
        "mode": "rate",
        "filter_name": (
            filter_name.decode() if isinstance(filter_name, bytes) else filter_name
        ),
        "cd_values_u32": [int(v) for v in cd_values],
        "raw_bytes_len": len(raw_bytes),
        "compressed_bytes_len": len(chunk_bytes),
        "notes": case.notes,
    }


def main() -> int:
    cases = build_cases()
    manifest: list[dict[str, Any]] = []
    print(f"generating {len(cases)} fixtures in {FIXTURE_DIR}")
    for c in cases:
        entry = write_fixture(c)
        manifest.append(entry)
        print(f"  {c.name}: raw={entry['raw_bytes_len']}B compressed={entry['compressed_bytes_len']}B")
    MANIFEST.write_text(json.dumps({"fixtures": manifest}, indent=2) + "\n")
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
