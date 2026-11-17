use core::num::NonZeroUsize;
use std::path::Path;

use hdf5_pure as facade;
use hdf5_pure_core::__private::FileRange;
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private as format;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageType;
use test_util::object_header::v1;
use test_util::object_header::v2::Header;

use super::Fixture;
use super::Form;
use crate::global_heap_fixture;

/// Unlinked residual content appended after fixed string and numeric datasets.
#[derive(Clone, Copy, Debug)]
pub enum ResidualGroupSource {
    /// A compact group with a `child` link to the string dataset `gone`.
    Gone,
    /// A compact group with a `child` link to the numeric dataset `keep`.
    Keep,
    /// A version 1 header with a zero-filled symbol-table record.
    Legacy,
    /// A header with version byte three and the remaining version 2 bytes unchanged.
    Malformed,
    /// A complete group header with a `child` link to `keep`, padded with zeros to the supplied
    /// byte length.
    Pad(NonZeroUsize),
    /// A group header with a `child` link to `keep`, repeated the supplied number of times.
    Repeat(NonZeroUsize),
    /// Residual bytes filled with 0x5a.
    Unknown,
    /// A group with a `child` link to one less than [`u64::MAX`].
    Unresolved,
    /// A group with a `child` link to `keep` and an additional unknown message type 0x7f.
    Unsupported,
}

impl ResidualGroupSource {
    /// Writes `gone` and `keep` with the selected unlinked residual bytes.
    ///
    /// The string dataset `gone` contains `["removed"]`, and the numeric dataset `keep` contains
    /// `[17]`. Their root links are written in that order.
    ///
    /// Returns the physical heap-parent address for `gone`, the residual start address, and the
    /// complete `keep` object-header range, in that order.
    ///
    /// # Panics
    ///
    /// Panics if padding is shorter than the complete generated group header, repetition
    /// overflows vector capacity, or construction, address conversion, or file I/O fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::num::NonZeroUsize;
    /// use hdf5_pure::File;
    /// use test_util_hdf5::__private::ResidualGroupSource;
    ///
    /// let dir = tempfile::tempdir()?;
    /// let path = dir.path().join("residual.h5");
    /// let length = NonZeroUsize::new(64).unwrap();
    /// let (parent, residual, keeper) = ResidualGroupSource::Pad(length).write(&path);
    /// assert!(parent < keeper.start().get());
    /// assert_eq!(keeper.end().get(), residual);
    /// assert_eq!(std::fs::metadata(&path)?.len(), residual + 64);
    /// let file = File::open(&path)?;
    /// assert_eq!(file.root().datasets()?, ["gone", "keep"]);
    /// assert_eq!(file.dataset("gone")?.read_string()?, ["removed"]);
    /// let keep = file.dataset("keep")?;
    /// assert_eq!(keep.read_i32()?, [17]);
    /// drop(keep);
    /// file.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write(self, path: &Path) -> (u64, u64, FileRange) {
        let extra = self;
        let mut fixture = Fixture::new();
        let parent = fixture.collection(&[Object::new(7, b"removed")]);
        let records = Form::ContiguousDataset.messages(
            &mut fixture,
            &global_heap_fixture::vlen_string(),
            &global_heap_fixture::reference(7, parent, 7),
        );
        let gone = fixture.append_header(records);
        fixture.link("gone", gone);
        let records = Form::CompactDataset.messages(
            &mut fixture,
            &facade::make_i32_type(),
            &17_i32.to_le_bytes(),
        );
        let keep = fixture.append_header(records);
        let keeper = FileRange::new(
            PhysicalAddress::new(keep),
            u64::try_from(fixture.image.as_bytes().len()).unwrap() - keep,
        )
        .unwrap();
        fixture.link("keep", keep);
        let mut records = vec![
            Message::new(
                MessageType::LINK_INFO,
                &format::LinkInfoMessage {
                    max_creation_order: None,
                    fractal_heap_address: None,
                    btree_name_index_address: None,
                    btree_creation_order_address: None,
                }
                .serialize(format::OffsetWidth::Eight)
                .unwrap(),
            ),
            global_heap_fixture::hard_link(
                "child",
                match extra {
                    Self::Gone => gone,
                    Self::Unresolved => u64::MAX - 1,
                    _ => keep,
                },
            ),
        ];
        if matches!(extra, Self::Unsupported) {
            records.push(Message::new(MessageType(0x7f), &[]));
        }
        let mut bytes = if matches!(extra, Self::Legacy) {
            v1::Header::new()
                .messages(vec![Message::new(MessageType::SYMBOL_TABLE, &[0; 16])])
                .build()
        } else {
            Header::new().messages(records).build()
        };
        match extra {
            Self::Unknown => bytes.fill(0x5a),
            Self::Malformed => bytes[4] = 3,
            Self::Pad(length) => {
                assert!(
                    length.get() >= bytes.len(),
                    "residual padding cannot shorten the header"
                );
                bytes.resize(length.get(), 0);
            }
            Self::Repeat(count) => bytes = bytes.repeat(count.get()),
            _ => {}
        }
        let group = u64::try_from(fixture.image.append(&bytes)).unwrap();
        fixture.write(path);
        (parent, group, keeper)
    }
}

#[cfg(test)]
mod tests;
