use hdf5_pure_core::MetadataBytes;

#[derive(Clone, Copy, Debug)]
pub(super) enum Backing {
    Borrowed,
    Owned,
}

pub(super) fn bytes(backing: Backing, offset: u64, data: &[u8]) -> MetadataBytes<'_> {
    match backing {
        Backing::Borrowed => MetadataBytes::from_slice(offset, data),
        Backing::Owned => MetadataBytes::from_vec(offset, data.to_vec()),
    }
    .unwrap()
}
