# CI images

Set the stable compiler in `rust-toolchain.toml` and the Miri nightly in `scripts/nightly.txt`.
CI derives Matio and Miri image tags from those pins and uses `latest` for the cross images.
Existing PR jobs build and test changed image definitions, and merged construction changes publish
automatically to the same mutable tags. Consumers build locally if a published image cannot be pulled.
