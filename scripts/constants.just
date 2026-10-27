# The dated nightly scripts/nightly.txt pins for the checks that need nightly.
NIGHTLY := trim(read(source_directory() / "nightly.txt"))
# The external interoperability test package.
CROSSCHECK := "hdf5-pure-crosscheck"
# Every workspace crate but the crosschecks, for the workspace-wide recipes.
WORKSPACE := "--workspace --exclude " + CROSSCHECK
# The crates that build without `std`, which the portability recipes check on every target.
PORTABLE := "-p hdf5-pure -p hdf5-pure-core -p hdf5-pure-filter -p hdf5-pure-format -p hdf5-pure-space"
# The workspace crates that publish to crates.io, as `-p` arguments.
PUBLISHED := "-p hdf5-pure -p hdf5-pure-core -p hdf5-pure-filter -p hdf5-pure-format -p hdf5-pure-space"
# Every documented public feature on the facade, including the default features.
PUBLIC_FEATURES := "std,checksum,deflate,fast-deflate,serde,ndarray,num-complex,provenance,zfp"
# The public facade features whose dependency graphs remain `no_std`-compatible.
NO_STD_PUBLIC_FEATURES := "checksum,provenance,zfp"
# The facade's public features the full recipes enable beside its defaults.
FULL_PUBLIC_FEATURES := "checksum,fast-deflate,ndarray,num-complex,provenance,serde,zfp"
# The complete feature sets for local tests, doctests, and CI shards.
FEATURE_TEST_CONFIGURATIONS := "std,deflate default," + replace(FULL_PUBLIC_FEATURES, ",", " default,")
# The same features on the facade, as cargo arguments for a run over several packages.
FULL_PUBLIC := "--features " + replace_regex(FULL_PUBLIC_FEATURES, '[^,]+', 'hdf5-pure/$0')
