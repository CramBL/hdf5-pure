The `parse_file` corpus contains small, valid HDF5 fixtures copied from
`tests/data/` so mutations start from realistic superblocks, object headers,
attributes, groups, and chunk metadata. CI also passes this corpus to
`streaming_differential`.
