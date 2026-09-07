#!/usr/bin/env python3
"""Generate `struct_array_v73.mat`, a synthetic MATLAB v7.3 file exercising the
struct-*array* on-disk layout (issue #127).

This is NOT real-MATLAB output (unlike `tests/data/matlab/`). It is built
with h5py to follow MATLAB's documented v7.3 struct-array convention exactly, as
implemented by the `foreverallama/matio` writer (`matwriter7.write_struct_array`):

  * a struct array is a `MATLAB_class="struct"` group with a `MATLAB_fields`
    attribute listing its fields;
  * each field is a dataset of HDF5 object references (one per array element),
    stored column-major (`refs.T`), carrying NO `MATLAB_class` of its own;
  * each reference points at a dataset under `#refs#` holding that element's
    field value, with the value's own `MATLAB_class`.

A scalar (1x1) struct, by contrast, stores its fields as direct value datasets;
`scalar` below pins that the reader still treats those as a struct, not an array.

Regenerate with:  python3 tests/data/h5py/mat/gen_struct_array.py
"""

import sys
import numpy as np
import h5py


def add_class(obj, name):
    obj.attrs.create("MATLAB_class", np.bytes_(name))


def add_utf16(obj):
    obj.attrs.create("MATLAB_int_decode", np.int32(4))  # UTF-16 hint


def set_fields(group, fields):
    dt = h5py.special_dtype(vlen=np.dtype("S1"))
    mf = np.empty((len(fields),), dtype=dt)
    for i, fn in enumerate(fields):
        mf[i] = np.array([c.encode("ascii") for c in fn], dtype="S1")
    group.attrs.create("MATLAB_fields", mf)


class Gen:
    def __init__(self, f):
        self.f = f
        self.refs = f.require_group("#refs#")
        self.n = 0

    def _name(self):
        n = "r%d" % self.n
        self.n += 1
        return n

    # --- value writers under #refs#, returning the dataset (for its .ref) ---
    def double_scalar(self, v):
        d = self.refs.create_dataset(self._name(), data=np.array([[v]], dtype="<f8"))
        add_class(d, "double")
        return d

    def char(self, s):
        arr = np.array([[ord(c) for c in s]], dtype="<u2")  # MATLAB 1xN row
        d = self.refs.create_dataset(self._name(), data=arr.T)
        add_class(d, "char")
        add_utf16(d)
        return d

    def double_rowvec(self, vals):
        arr = np.array([vals], dtype="<f8")  # 1xN row
        d = self.refs.create_dataset(self._name(), data=arr.T)
        add_class(d, "double")
        return d

    def scalar_struct(self, fields_values):
        """A nested scalar struct written under #refs#."""
        g = self.refs.create_group(self._name())
        add_class(g, "struct")
        for name, dset in fields_values:
            # Re-link: write value then hard-link under the struct group's name.
            g[name] = dset  # h5py hard link
        set_fields(g, [n for n, _ in fields_values])
        return g

    def struct_array(self, parent, var, shape, fields, elem):
        """elem(field, (r, c)) -> a #refs# dataset/group whose .ref is stored."""
        g = parent.create_group(var)
        add_class(g, "struct")
        total = int(np.prod(shape))
        for field in fields:
            refs = np.empty(shape, dtype=h5py.ref_dtype)
            for flat in range(total):
                idx = np.unravel_index(flat, shape)
                refs[idx] = elem(field, idx).ref
            g.create_dataset(field, data=refs.T, dtype=h5py.ref_dtype)
        set_fields(g, list(fields))
        return g


LETTERS = "abcdef"
VEC = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.0]


def main(out):
    with h5py.File(out, "w", userblock_size=512) as f:
        g = Gen(f)

        # `row`: 1x6 struct array (the issue's example, row orientation).
        def row_elem(field, idx):
            c = idx[1]
            if field == "fieldA":
                return g.double_scalar(float(c + 1))
            if field == "fieldB":
                return g.char(LETTERS[c])
            return g.double_rowvec(VEC)

        g.struct_array(f, "row", (1, 6), ["fieldA", "fieldB", "fieldC"], row_elem)

        # `col`: 6x1 struct array (column orientation; flattens identically).
        def col_elem(field, idx):
            r = idx[0]
            if field == "fieldA":
                return g.double_scalar(float(r + 1))
            if field == "fieldB":
                return g.char(LETTERS[r])
            return g.double_rowvec(VEC)

        g.struct_array(f, "col", (6, 1), ["fieldA", "fieldB", "fieldC"], col_elem)

        # `grid`: 2x3 struct array; `id` = r*10 + c pins row-major ordering.
        def grid_elem(field, idx):
            r, c = idx
            if field == "id":
                return g.double_scalar(float(r * 10 + c))
            return g.char(LETTERS[r * 3 + c])

        g.struct_array(f, "grid", (2, 3), ["id", "tag"], grid_elem)

        # `nested`: 1x2 struct array whose elements contain a nested scalar
        # struct field (`inner.p`), to exercise composition through references.
        def nested_elem(field, idx):
            i = idx[1]
            if field == "fieldA":
                return g.double_scalar(float(i + 1))
            p = g.double_scalar(float(i * 100))
            return g.scalar_struct([("p", p)])

        g.struct_array(f, "nested", (1, 2), ["fieldA", "inner"], nested_elem)

        # `scalar`: a 1x1 struct (direct value fields) — regression guard that a
        # scalar struct is not misdetected as a struct array.
        sg = f.create_group("scalar")
        add_class(sg, "struct")
        d = sg.create_dataset("fieldA", data=np.array([[1.0]], dtype="<f8"))
        add_class(d, "double")
        ch = sg.create_dataset("fieldB", data=np.array([[ord("a")]], dtype="<u2").T)
        add_class(ch, "char")
        add_utf16(ch)
        vc = sg.create_dataset("fieldC", data=np.array([VEC], dtype="<f8").T)
        add_class(vc, "double")
        set_fields(sg, ["fieldA", "fieldB", "fieldC"])

    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "struct_array_v73.mat")
