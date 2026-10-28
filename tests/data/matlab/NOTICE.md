# Vendored real-MATLAB `.mat` v7.3 fixtures

These files retain MATLAB v7.3 output and two deliberately corrupted derivatives.
All headers identify `PCWIN64` and HDF5 schema 1.00. The MATLAB release is
unrecorded except for the repository-authored unset-field case below.

Retain these frozen bytes as external MCOS, value, and corruption evidence.
Writing equivalent decoded values into a new file would discard the original
object metadata or malformed conditions. Ordinary tests never regenerate these files.

## Consumers and retained observations

| File | Direct consumer under `tests/main/` unless qualified | Invariant |
|---|---|---|
| `matjl_struct_table_datetime.mat` | `mat_opaque_real_read.rs` | Datetime double-double residual in version 1 compound `{real, imag}` storage |
| `test_corrupted_mcos_object_metadata.mat` | `mat_opaque_real_read.rs` | Out-of-range MCOS object ID returns an error without panic |
| `test_corrupted_subsystem.mat` | `mat_opaque_real_read.rs` | The reader returns success or an error without panic |
| `test_enum_v73.mat` | `mat_enum_read.rs`, `src/superblock.rs` | Scalar enums, uint32 bases, arrays, nested enums, and user-block-relative parsing |
| `test_maps_v73.mat` | `mat_containers_map_read.rs` | Map and dictionary keys and values, including nested mappings |
| `test_string_v73.mat` | `mat_opaque_real_read.rs`, `edit_userblock.rs`, `edit_userblock_chunked.rs`, `crates/crosscheck/tests/main/edit_userblock.rs` | MCOS strings and user-block preservation |
| `test_struct_unset_field_v73.mat` | `mat_empty_marker_conformance.rs` | Unset structure field refers to the canonical empty placeholder |
| `test_tables_v73.mat` | `mat_opaque_real_read.rs`, `mat_table_read.rs`, `mat_nested_mcos_read.rs` | Tables, timetables, categorical and nested MCOS columns, row and variable metadata |
| `test_time_v73.mat` | `mat_opaque_real_read.rs` | Datetime and duration values, dimensions, classes, and timezone metadata |

All files also participate in the directory scans in
`mat_empty_marker_conformance.rs` and `mat_refs_h5path.rs`. Those scans skip files
that fail to open, they do not require rejection of each corrupted input.

The two corrupted files are MATLAB-derived regressions supplied by the pinned
upstream below. Their exact mutation operations and original MATLAB releases
are unrecorded locally.

## Source

Vendored from the **`foreverallama/matio`** project (the `mat-io` Python
library), directory `tests/data/`:

- Repository: <https://github.com/foreverallama/matio>
- Commit: `77a578b6605f90035a4030b4dbce0cbe69e0a3d5`
- Files: `test_string_v73.mat`, `test_time_v73.mat`, `test_tables_v73.mat`,
  `test_enum_v73.mat`, `test_maps_v73.mat`, `test_corrupted_subsystem.mat`,
  `test_corrupted_mcos_object_metadata.mat`

The expected decoded values for the valid files are transcribed from that project's MATLAB generator scripts
(`tests/data/generators/*.m`) and its independent pytest oracle
(`tests/test_datetime.py`, `test_duration.py`, `test_categorical.py`,
`test_matstring.py`, `test_enum.py`, `test_containermap.py`).

### Also vendored: `matjl_struct_table_datetime.mat`

From the **`JuliaIO/MAT.jl`** project (`test/v7.3/struct_table_datetime.mat`),
real MATLAB output (`Platform: PCWIN64`):

- Repository: <https://github.com/JuliaIO/MAT.jl>
- Commit: `4e05a0169282b6b7d835569175f07dff14ddabad`
- License: MIT (see below)

The upstream `struct_table_datetime.mat` is retained locally as
`matjl_struct_table_datetime.mat`. Its datetime has a nonzero imaginary
sub-millisecond component stored as a version 1 compound `{real, imag}` of
`f64`. Preserve that external encoding for the datetime double-double and
compound-parser observations.

### Also: `test_struct_unset_field_v73.mat`

Written by MATLAB R2026a (`Platform: PCWIN64`) for this repository with:

```matlab
s(2).x = 1;
save('test_struct_unset_field_v73.mat', 's', '-v7.3');
```

Because `s(1).x` was never set, MATLAB stores it as a reference to the `canonical empty`
placeholder. The file is distributed under this crate's license.

## Licenses

`foreverallama/matio` is distributed under the BSD 3-Clause License and
`JuliaIO/MAT.jl` under the MIT License. Their copyright notices are retained
below as required.

```
BSD 3-Clause License

Copyright (c) 2025, foreverallama

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

```
The MIT License (MIT)

Copyright (c) 2012 Simon Kornblith and Timothy E. Holy

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
