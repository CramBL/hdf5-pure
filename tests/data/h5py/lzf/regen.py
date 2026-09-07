#!/usr/bin/env python3
# /// script
# requires-python = "==3.14.*"
# dependencies = [
#     "h5py==3.16.0",
#     "numpy==2.4.4",
# ]
# ///
"""Regenerate LZF crosscheck fixtures from h5py's built-in LZF filter, and
verify that h5py can read what hdf5-pure writes.

Phase 1 (write) produces a `manifest.json` plus, per fixture:
  <name>.raw.bin          raw uncompressed values (little-endian native width)
  <name>.compressed.bin   stored chunk bytes, only for single-chunk cases
                          whose pipeline is exactly [lzf] — the only streams
                          src/lzf_crosscheck.rs compares at the codec level
                          (raw bytes when h5py skipped the optional filter —
                          see `filter_mask` in the manifest)
  <name>.h5               the h5py-written file, read end-to-end by
                          tests/lzf_roundtrip.rs and handy for h5dump

Phase 2 (read back) points h5py at `pure_written.h5`, the committed file
*hdf5-pure* wrote, and checks h5py decodes every dataset and can write an
incompressible chunk back into it. This is the only check of the write
direction that a real reference implementation performs: the Rust suite
compares our filter pipeline against this manifest, but it has no h5py, and
our LZF stream is deliberately not byte-compared against liblzf's (any
conforming stream is valid). Run it whenever `src/lzf.rs` or the LZF branch of
`ChunkOptions::build_pipeline` changes. `pure_written.h5` is kept in step with
the writer by `pure_written_fixture_is_current` in tests/lzf_roundtrip.rs.

LZF ships with h5py itself (no hdf5plugin needed). h5py registers it as an
*optional* filter: when liblzf cannot shrink a chunk the chunk is stored raw
with its filter-mask bit set — the `u8_noise` case captures that path, which
hdf5-pure's writer never takes (it stores the grown stream instead). hdf5-pure
does record the same optional flag, which is what lets the phase-2 write-back
below succeed.

Run from the repo root with uv (resolves the pins above automatically):
    uv run tests/data/h5py/lzf/regen.py

Or in a venv pinned from the sibling requirements file:
    python -m venv tests/data/h5py/lzf/.venv
    tests/data/h5py/lzf/.venv/bin/pip install -r tests/data/h5py/lzf/requirements.txt
    tests/data/h5py/lzf/.venv/bin/python tests/data/h5py/lzf/regen.py

Pass --verify-only to run phase 2 alone and leave the fixtures untouched.
"""

from __future__ import annotations

import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parent
MANIFEST = FIXTURE_DIR / "manifest.json"

NP_DTYPE = {
    "u8": np.uint8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "f64": np.float64,
}


@dataclass
class Case:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: Any  # numpy array (any dtype; cast on write)
    chunks: tuple[int, ...] | None = None  # None = one chunk covering shape
    shuffle: bool = False
    notes: str = ""

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.data, dtype=NP_DTYPE[self.dtype]).reshape(self.shape)


def xorshift_noise(n: int) -> np.ndarray:
    """Deterministic incompressible bytes.

    Same xorshift constants and seed as the `round_trips` unit test in
    src/lzf.rs; the Rust integration tests consume this fixture's bytes
    directly rather than regenerating them.
    """
    out = np.empty(n, dtype=np.uint8)
    x = 0x2545_F491_4F6C_DD1D
    mask = (1 << 64) - 1
    for i in range(n):
        x = (x ^ (x << 13)) & mask
        x ^= x >> 7
        x = (x ^ (x << 17)) & mask
        out[i] = x & 0xFF
    return out


def build_cases() -> list[Case]:
    return [
        Case(
            name="i32_ramp",
            dtype="i32",
            shape=(1024,),
            data=np.arange(1024),
            notes="compressible monotonic ramp, lzf only",
        ),
        Case(
            name="u8_noise",
            dtype="u8",
            shape=(4096,),
            data=xorshift_noise(4096),
            notes="incompressible; h5py's optional LZF stores the chunk raw "
            "with the filter-mask bit set",
        ),
        Case(
            name="i16_rle",
            dtype="i16",
            shape=(2048,),
            data=np.repeat(np.arange(8), 256),
            notes="long constant runs (overlapping back-references)",
        ),
        # sqrt, not sin. np.sin is whatever the host libm computes: regenerating
        # this case on a second machine, with these same pinned h5py and numpy
        # versions, moved 29 of the 512 values by 1 ULP and rewrote the fixture.
        # sqrt is correctly rounded by IEEE-754 mandate, so a regen produces the
        # same bytes anywhere and a real change stands out in the diff.
        Case(
            name="f64_shuffle_lzf",
            dtype="f64",
            shape=(512,),
            data=np.sqrt(np.arange(512)),
            shuffle=True,
            notes="shuffle+lzf chain; end-to-end .h5 coverage only",
        ),
        Case(
            name="i64_multichunk",
            dtype="i64",
            shape=(300,),
            data=np.arange(300) * 7 - 500,
            chunks=(128,),
            notes="multi-chunk with partial edge chunk; end-to-end .h5 only",
        ),
        # Half noise, half zeros. The noise half forces liblzf to emit literal
        # runs at the encoding's maximum (ctrl == 31, 32 literals), which no
        # other fixture reaches — the longest literal run in the rest of the
        # set is 5 bytes, so an off-by-one in MAX_LITERAL_RUN that shortens the
        # run is invisible to them. The zero half keeps the chunk compressible
        # overall, so h5py applies the filter instead of skipping it.
        Case(
            name="u8_literal_runs",
            dtype="u8",
            shape=(4096,),
            data=np.concatenate(
                [xorshift_noise(2048), np.zeros(2048, dtype=np.uint8)]
            ),
            notes="maximum-length literal runs (ctrl == 31) followed by a "
            "compressible tail",
        ),
        # The only rank > 1 chunk in the set. cd_values[2] is a product over
        # element size and every chunk dimension, so a writer that folded over
        # just the first dimension — or confused the two factors — agrees with
        # every 1-D fixture and disagrees here (4 * 16 * 10 = 640).
        #
        # Constant rows rather than a ramp: at this size an i32 ramp does not
        # shrink, and h5py then stores the chunk raw, which would leave the
        # rank-2 stream undecoded by the crosscheck.
        Case(
            name="i32_rank2",
            dtype="i32",
            shape=(16, 10),
            data=np.repeat(np.arange(16), 10),
            notes="rank-2 chunk; pins cd_values[2] as a product over all dims",
        ),
    ]


def write_fixture(case: Case) -> dict[str, Any]:
    arr = case.to_numpy()
    h5_path = FIXTURE_DIR / f"{case.name}.h5"
    chunks = case.chunks or arr.shape

    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "v", data=arr, chunks=chunks, compression="lzf", shuffle=case.shuffle
        )

    with h5py.File(h5_path, "r") as f:
        d = f["v"]
        dcpl = d.id.get_create_plist()
        filters = [dcpl.get_filter(i) for i in range(dcpl.get_nfilters())]
        lzf = next(flt for flt in filters if flt[0] == 32000)
        _, lzf_flags, lzf_cd_values, _ = lzf
        # The chunk mask is a measured, per-chunk fact: record it only for
        # single-chunk cases rather than fabricating a whole-dataset value.
        if chunks == arr.shape:
            filter_mask, chunk_bytes = d.id.read_direct_chunk((0,) * arr.ndim)
        else:
            filter_mask, chunk_bytes = None, None

    raw_bytes = arr.tobytes(order="C")
    (FIXTURE_DIR / f"{case.name}.raw.bin").write_bytes(raw_bytes)
    # Only a bare-lzf stream is codec-comparable; chained pipelines are
    # exercised end-to-end through the .h5 instead.
    if chunk_bytes is not None and [int(f[0]) for f in filters] == [32000]:
        (FIXTURE_DIR / f"{case.name}.compressed.bin").write_bytes(chunk_bytes)
    else:
        chunk_bytes = None

    return {
        "name": case.name,
        "dtype": case.dtype,
        "shape": list(case.shape),
        "chunk_shape": list(chunks),
        "filters": [int(flt[0]) for flt in filters],
        "lzf_flags": int(lzf_flags),
        "cd_values_u32": [int(v) & 0xFFFFFFFF for v in lzf_cd_values],
        "filter_mask": None if filter_mask is None else int(filter_mask),
        "raw_bytes_len": len(raw_bytes),
        "compressed_bytes_len": len(chunk_bytes) if chunk_bytes is not None else None,
        "notes": case.notes,
    }


PURE_WRITTEN = FIXTURE_DIR / "pure_written.h5"

# Datasets in pure_written.h5, mirroring `build_pure_written` in
# tests/lzf_roundtrip.rs. Keep the two in step: the Rust test owns the file's
# contents, this table only says what h5py should find there.
PURE_EXPECTED = {
    "plain_i32": lambda: np.arange(1024, dtype=np.int32),
    # sqrt, not sin: pure_written.h5 is compared byte for byte by a Rust test
    # that runs on three platforms, and only correctly-rounded operations are
    # bit-identical across libms. See `build_pure_written` in lzf_roundtrip.rs.
    "shuffle_f64": lambda: np.sqrt(np.arange(512, dtype=np.float64)),
    "multichunk_i64": lambda: np.arange(1000, dtype=np.int64) * 7 - 500,
    "incompressible_u8": lambda: np.frombuffer(
        (FIXTURE_DIR / "u8_noise.raw.bin").read_bytes(), dtype=np.uint8
    ),
}


def verify_pure_written() -> None:
    """Check h5py reads hdf5-pure's own LZF output, and can write back into it.

    Two distinct claims. Reading proves our streams and cd_values are what
    liblzf's decoder expects. The write-back proves we recorded LZF as
    *optional*: h5py's filter returns failure on a chunk liblzf cannot shrink,
    and only the optional flag lets HDF5 store that chunk raw instead of
    failing the write. With a mandatory LZF this raises, and hard.
    """
    if not PURE_WRITTEN.exists():
        raise SystemExit(
            f"{PURE_WRITTEN.name} is missing. It is committed; regenerate it via "
            "`build_pure_written` in tests/lzf_roundtrip.rs."
        )

    with h5py.File(PURE_WRITTEN, "r") as f:
        missing = set(PURE_EXPECTED) - set(f)
        if missing:
            raise SystemExit(f"{PURE_WRITTEN.name}: datasets missing: {sorted(missing)}")
        for name, expected_fn in PURE_EXPECTED.items():
            d = f[name]
            ids = [d.id.get_create_plist().get_filter(i)[0] for i in range(d.id.get_create_plist().get_nfilters())]
            if 32000 not in ids:
                raise SystemExit(f"{name}: not LZF-compressed (filters {ids})")
            got = d[...]
            expected = expected_fn()
            if not np.array_equal(got, expected):
                raise SystemExit(f"{name}: h5py decoded hdf5-pure's LZF stream incorrectly")
            print(f"  read  {name:<18} {got.dtype} x{got.size} OK")

    # h5py writing an incompressible chunk into our file: the optional-flag path.
    #
    # Two traps here, both of which this check walked into before they were
    # closed. The data written must *differ* from what the dataset already
    # holds, or a write HDF5 dropped on the floor still reads back equal. And
    # the failure does not raise: HDF5 defers the chunk write, so the error
    # surfaces as a RuntimeError inside h5py's `__dealloc__`, which Python
    # reports through `sys.unraisablehook` and otherwise ignores. A check that
    # only wraps the assignment in try/except sees a clean run and prints OK.
    scratch = FIXTURE_DIR / "pure_written.writeback.tmp.h5"
    scratch.write_bytes(PURE_WRITTEN.read_bytes())
    unraisable: list[Any] = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = unraisable.append
    try:
        original = PURE_EXPECTED["incompressible_u8"]()
        # Still incompressible (XOR by a constant and a rotation both preserve
        # the entropy), but not the bytes already in the file.
        altered = np.roll(original, 1) ^ np.uint8(0xA5)
        assert not np.array_equal(altered, original), "write-back data must differ"

        with h5py.File(scratch, "r+") as f:
            f["incompressible_u8"][...] = altered
        gc.collect()  # force any deferred __dealloc__ before inspecting the hook
        if unraisable:
            errors = "; ".join(repr(u.exc_value) for u in unraisable)
            raise SystemExit(
                "write-back: h5py failed to write an incompressible chunk into "
                f"hdf5-pure's file ({errors}).\nThis is what a *mandatory* LZF "
                "does: liblzf declines the chunk and HDF5 has no permission to "
                "store it raw. hdf5-pure must record LZF with flags=1."
            )
        with h5py.File(scratch, "r") as f:
            if not np.array_equal(f["incompressible_u8"][...], altered):
                raise SystemExit(
                    "write-back: h5py's chunk did not read back — the write was "
                    "silently dropped"
                )
        print("  write incompressible chunk back through h5py OK (LZF is optional)")
    finally:
        sys.unraisablehook = previous_hook
        scratch.unlink(missing_ok=True)


def main() -> None:
    verify_only = "--verify-only" in sys.argv[1:]

    if not verify_only:
        entries = [write_fixture(case) for case in build_cases()]
        manifest = {
            "generator": f"h5py {h5py.__version__}, numpy {np.__version__}",
            "fixtures": entries,
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {len(entries)} fixtures to {FIXTURE_DIR}")

    print(f"verifying {PURE_WRITTEN.name} against h5py {h5py.__version__}:")
    verify_pure_written()
    print("h5py reads hdf5-pure's LZF output")


if __name__ == "__main__":
    main()
