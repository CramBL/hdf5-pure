# HDF5 1.8-era read fixtures

`v1_superblock.h5` and `v2_superblock.h5` are written by **HDF5 1.8.23**, the
last 1.8 release, and committed as ground-truth read fixtures for
`tests/c_1_8_read_compat.rs` and the `parse_v1_against_a_c_written_superblock`
unit test in `src/superblock.rs`.

They are produced by `regen.c` in this directory, built and run by `regen.sh`
against the 1.8.23 install that `scripts/check_hdf5_compat.py` leaves under
`tmp/hdf5-compat-check/install/`. Nothing in `cargo test` invokes either: the fixtures are
committed, so the tests reading them are ordinary tests with no external
dependency.

## Why committed bytes, given the crosschecks already exist

These are not the first coverage of either format, and are not meant to replace
what is there:

- `tests/owned_swmr_crosscheck.rs` asks libhdf5 for a version 1 superblock
  (`istore_k(33)`) and checks the parsed chunk B-tree K and status flags.
- `tests/edit_crosscheck.rs` does the same for a version 2 superblock.

What both cost is the `hdf5-metno` dev-dependency, which requires 64-bit
pointers. Every file using it therefore opens with
`#![cfg(not(target_pointer_width = "32"))]` and compiles out on the i686 target
— which is where address arithmetic is most likely to be wrong, and the reason
the crate has a 32-bit CI job at all. Reading committed bytes needs no
dev-dependency, so `tests/c_1_8_read_compat.rs` is in that job's target list.

The committed corpus also had nothing at these versions. Of the 80 tracked
`.h5`/`.mat` fixtures before these two, **69 were superblock 0 and 11 were
superblock 3**.

And one thing here exists nowhere else: `v2_superblock.h5` is a **version 2
superblock carrying a version 1 B-tree chunk index**. This crate's writer cannot
produce that pairing — it refuses chunked storage under a 1.8 bound, because the
only chunk indices it writes arrived in 1.10 — so no round trip through it can
stand in for the file.

## What `v1_superblock.h5` pins

The version 1 superblock adds **one** field to version 0: the chunk B-tree K.
(Version 0 already carries the two group K values.) It sits beside the status
flags, and a defect fixed in 0.33.0 read the two from each other's offsets.

That mattered because the status flags are the field `File::open` consults to
refuse a file another writer holds. It could not misfire on a version 1 file
itself — `file_lock::check_status_flags` returns early below superblock version
3 — so the defect was a wrong answer from `File::superblock()`, not a wrongly
refused open. The parse is what these fixtures pin.

`src/superblock.rs` tested `parse_v1` against `build_v1_bytes`, which the same
hand wrote from the same reading of the specification; the two agree about the
field order whether or not that order is right. Laying the bytes out the wrong
way in *both* leaves both hand-built tests passing. It does not survive the
crosscheck above, and it does not survive this fixture either — the difference
being that this one still runs when `hdf5-metno` is unavailable.

The file is written with `H5Pset_sym_k(8, 16)` and `H5Pset_istore_k(64)`, so all
three K values differ from one another and from 1.8.23's defaults (4 leaf, 16
internal, 32 chunk), and any permutation of the three reads back wrong.

## Contents

Both files hold the same objects, so a failure in one and not the other names
the format:

- `/values` — contiguous `f64[4]`, with a `units` string attribute
- `/chunked` — `i32[1000]` in 100-element chunks, deflate level 6. Written under
  1.8 bounds in both files, so both are indexed by a version 1 B-tree.
- `/grp` — a group with a `tag` attribute, holding `/grp/inner`. A v1 symbol
  table in `v1_superblock.h5`, a link-message group in `v2_superblock.h5` —
  a consequence of the libver bound `regen.c` uses, not of the superblock
  version.
- `/` — a `root_attr` string attribute

## Regenerating

`regen.sh` refuses to run against anything but HDF5 1.8.x, and checks the
superblock version of each file it wrote. Both matter: `H5F_LIBVER_LATEST` in
`regen.c` means "the 1.8 format" only because it is compiled against 1.8, and
against 1.10 or newer the same source writes a version 3 superblock into
`v2_superblock.h5` and a version 2 one into `v1_superblock.h5`.

Rerun it only to change what the fixtures contain. The committed files are the
ground truth the tests read; regenerating them for no reason replaces measured
bytes with freshly measured bytes and reviews as a diff nobody can check.

## Licensing

Nothing here is vendored from the HDF5 distribution. `regen.c` is this project's
own code, and the `.h5` files are its output. HDF5 1.8.23 is used as a build
tool, the way a compiler is; it is not redistributed, and its own test files are
deliberately not copied — they exercise libhdf5 features this crate does not
support, so most would be refusals rather than coverage.
