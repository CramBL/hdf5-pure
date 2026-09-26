import re

from hdf5_pure_scripts import nightly


def test_the_pinned_toolchain_is_a_dated_nightly():
    assert re.fullmatch(r"nightly-\d{4}-\d{2}-\d{2}", nightly.toolchain())
