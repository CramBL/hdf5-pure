use hdf5_pure::{
    File, FileAccessProperties, FileBuilder, FormatError, MetadataBytes, MetadataCacheConfig,
    Source,
};

use rstest::rstest;

struct BorrowedSource<T>(T);

impl<T: AsRef<[u8]>> Source for BorrowedSource<T> {
    fn len(&self) -> u64 {
        self.0.as_ref().len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<(), FormatError> {
        let offset = usize::try_from(offset).unwrap();
        buf.copy_from_slice(&self.0.as_ref()[offset..offset + buf.len()]);
        Ok(())
    }

    fn read_metadata_at(&self, _offset: u64, _len: usize) -> Result<Vec<u8>, FormatError> {
        Err(FormatError::Source("unexpected owned metadata read".into()))
    }

    fn read_metadata(&self, offset: u64, len: usize) -> Result<MetadataBytes<'_>, FormatError> {
        let start = usize::try_from(offset).unwrap();
        MetadataBytes::from_slice(offset, &self.0.as_ref()[start..start + len])
    }
}

#[test]
fn source_metadata_owners_preserve_borrows_through_trait_objects() {
    let data = [1, 2, 3, 4];
    let source = BorrowedSource(&data);
    let object: &dyn Source = &source;
    let bytes: hdf5_pure_core::MetadataBytes<'_> = object.read_metadata(1, 2).unwrap();
    assert_eq!(object.metadata_origin().unwrap(), 0);
    assert_eq!(bytes.offset(), 1);
    assert_eq!(bytes.as_ref(), &[2, 3]);
    assert_eq!(bytes.as_ptr(), data[1..].as_ptr());
}

#[test]
fn owned_metadata_clones_keep_bytes_after_the_original_is_dropped() {
    let original = MetadataBytes::from_vec(512, vec![1, 2, 3]).unwrap();
    let retained = original.clone();
    let pointer = original.as_ptr();
    drop(original);
    assert_eq!(retained.offset(), 512);
    assert_eq!(retained.as_ref(), &[1, 2, 3]);
    assert_eq!(retained.as_ptr(), pointer);
    let owned = retained.into_vec();
    assert_eq!(owned, vec![1, 2, 3]);
    assert_eq!(owned.as_ptr(), pointer);
}

#[rstest]
#[case::uncached(false)]
#[case::cache_enabled(true)]
fn file_parsers_use_source_metadata_owners(#[case] cache_enabled: bool) {
    let mut builder = FileBuilder::new();
    builder
        .create_dataset("values")
        .with_i32_data(&[1, 2, 3, 4])
        .with_shape(&[4])
        .with_chunks(&[2]);
    let source = BorrowedSource(builder.finish().unwrap());
    let file = if cache_enabled {
        File::from_source_with_options(
            source,
            FileAccessProperties::new().with_metadata_cache(MetadataCacheConfig::new(4096)),
        )
    } else {
        File::from_source(source)
    }
    .unwrap();
    assert_eq!(
        file.dataset("values").unwrap().read_i32().unwrap(),
        vec![1, 2, 3, 4]
    );
    if cache_enabled {
        let stats = file.metadata_cache_stats().unwrap();
        assert!(stats.misses() > 0);
        assert_eq!((stats.hits(), stats.entries(), stats.bytes()), (0, 0, 0));
    } else {
        assert_eq!(file.metadata_cache_stats(), None);
    }
}
