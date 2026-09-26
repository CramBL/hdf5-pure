# The dated nightly scripts/nightly.txt pins for the checks that need nightly.
NIGHTLY := trim(read(source_directory() / "nightly.txt"))
# The package that links the C library, which only the interop recipes build.
CROSSCHECK := "hdf5-pure-crosscheck"
# Every workspace crate but the crosschecks, for the workspace-wide recipes.
WORKSPACE := "--workspace --exclude " + CROSSCHECK
# The crates that build without `std`, which the portability recipes check on every target.
PORTABLE := "-p hdf5-pure -p hdf5-pure-core -p hdf5-pure-filter -p hdf5-pure-format"
# The workspace crates that publish to crates.io, as `-p` arguments.
PUBLISHED := "-p hdf5-pure -p hdf5-pure-core -p hdf5-pure-filter -p hdf5-pure-format"
# The facade's public features the full recipes enable beside its defaults.
FULL_PUBLIC_FEATURES := "checksum,fast-deflate,ndarray,num-complex,provenance,serde,zfp"
# The same features on the facade, as cargo arguments for a run over several packages.
FULL_PUBLIC := "--features " + replace_regex(FULL_PUBLIC_FEATURES, '[^,]+', 'hdf5-pure/$0')
