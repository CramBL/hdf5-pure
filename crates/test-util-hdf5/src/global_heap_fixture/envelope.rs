use std::path::Path;

use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use test_util::global_heap;
use test_util::global_heap::MIN_COLLECTION_SIZE;
use test_util::global_heap::Object;
use test_util::object_header::v2::Header;
use test_util::widths::Widths;

use super::Fixture;
use super::Form;
use crate::global_heap_fixture;

/// Physical addresses and residual storage for healthy and damaged collection sources.
pub struct EnvelopeSource {
    /// The healthy collection's physical start.
    pub healthy: u64,
    /// The damaged collection's physical start.
    pub damaged: u64,
    /// The object-header address of the `gone` dataset, which refers to the healthy collection.
    pub gone: u64,
    /// The optional unexplained residual range.
    pub opaque: Option<FileRange>,
}

impl EnvelopeSource {
    /// Writes string owners of healthy and damaged heap collections with a numeric keeper.
    ///
    /// `damage` selects the collection damage and any additional source restriction. Records both
    /// parent addresses, the `gone` dataset's header address, and any opaque residual range in the
    /// returned source.
    ///
    /// # Panics
    ///
    /// Panics if construction, address conversion, or writing the fixture fails.
    pub fn write(path: &Path, damage: Damage) -> Self {
        let mut fixture = Fixture::new();
        let mut damaged =
            fixture.collection(&[Object::new(7, b"before"), Object::new(7, b"duplicate")]);
        let healthy = fixture.collection(&[Object::new(7, b"removed")]);
        let records = Form::CompactDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(6, damaged, 7),
        );
        let bad_header = fixture.append_header(records);
        fixture.link("bad", bad_header);
        let bad_header_length = u64::try_from(fixture.image.as_bytes().len()).unwrap() - bad_header;
        let records = Form::ContiguousDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(7, healthy, 7),
        );
        let gone = fixture.append_header(records);
        fixture.link("gone", gone);
        let records = Form::CompactDataset.messages(
            &mut fixture,
            &hdf5_pure::make_i32_type(),
            &17_i32.to_le_bytes(),
        );
        fixture.object("keep", records);
        if matches!(damage, Damage::BeyondSource) {
            fixture.image.place(
                usize::try_from(damaged).unwrap(),
                &vec![0; MIN_COLLECTION_SIZE],
            );
            damaged = u64::try_from(fixture.image.append(&global_heap::collection(
                &[Object::new(7, b"before"), Object::new(7, b"duplicate")],
                Widths::EIGHT,
            )))
            .unwrap();
            let records = Form::CompactDataset.messages(
                &mut fixture,
                &global_heap_fixture::vlen_string(),
                &global_heap_fixture::reference(6, damaged, 7),
            );
            let header = Header::new().messages(records).build();
            assert_eq!(u64::try_from(header.len()).unwrap(), bad_header_length);
            fixture
                .image
                .place(usize::try_from(bad_header).unwrap(), &header);
        }
        let start = usize::try_from(damaged).unwrap();
        let mut opaque = None;
        match damage {
            Damage::Version => fixture
                .image
                .place(start + global_heap::VERSION_OFFSET, &[2]),
            Damage::Overlap => fixture.image.place(
                start + global_heap::SIZE_OFFSET,
                &u64::try_from(2 * MIN_COLLECTION_SIZE)
                    .unwrap()
                    .to_le_bytes(),
            ),
            Damage::BeyondSource => fixture.image.place(
                start + global_heap::SIZE_OFFSET,
                &u64::try_from(OVERSIZED_COLLECTION).unwrap().to_le_bytes(),
            ),
            Damage::Unknown => {
                let start = fixture.image.append(&OPAQUE_BYTES);
                opaque = Some(
                    FileRange::new(
                        PhysicalAddress::new(u64::try_from(start).unwrap()),
                        u64::try_from(OPAQUE_BYTES.len()).unwrap(),
                    )
                    .unwrap(),
                );
            }
            Damage::Records => {}
        }
        fixture.write(path);
        Self {
            healthy,
            damaged,
            gone,
            opaque,
        }
    }
}

/// Source restrictions for a collection with duplicate member indices.
#[derive(Clone, Copy, Debug)]
pub enum Damage {
    /// Declares a collection allocation past the source end.
    BeyondSource,
    /// Extends the damaged allocation across the healthy collection.
    Overlap,
    /// Preserves the duplicate member records without another source restriction.
    Records,
    /// Appends unexplained nonzero residual bytes.
    Unknown,
    /// Uses unsupported collection version 2.
    Version,
}

/// Opaque residual bytes appended for [`Damage::Unknown`].
pub const OPAQUE_BYTES: [u8; 17] = [0x5a; 17];

/// Declared allocation length used for source-extent rejection.
const OVERSIZED_COLLECTION: usize = 16 * MIN_COLLECTION_SIZE;
