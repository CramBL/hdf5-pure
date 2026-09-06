# /// script
# requires-python = ">=3.9,<3.12"
# dependencies = ["h5py>=3.6,<3.10", "numpy<2"]
#
# [tool.uv]
# no-binary-package = ["h5py"]
# ///
"""`verify_fixtures.py` for HDF5 1.8, which needs an older h5py.

h5py dropped HDF5 below 1.10.7 after its 3.9 release, which predates NumPy 2
and Python 3.12. The checks are the same, and only the pins differ, so this
file holds nothing but them.

The floor is 3.9 rather than 3.11 because conda-forge's hdf5 1.8.20 pins
zlib 1.2.11, which no newer python build accepts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_fixtures import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
