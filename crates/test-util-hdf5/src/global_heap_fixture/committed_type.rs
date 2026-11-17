use std::path::Path;

use hdf5_pure::Datatype;
use hdf5_pure::DatatypeByteOrder;
use hdf5_pure::FloatingPointType;
use test_util::datatype;
use test_util::global_heap::Object;
use test_util::object_header::Message;
use test_util::object_header::MessageFlags;
use test_util::object_header::MessageType;

use super::Fixture;
use super::Form;
use crate::global_heap_fixture;

/// The count record, hard links, and optional dataset attribute of a committed-datatype fixture.
///
/// Explicit counts are written unchanged, including zero.
pub enum CommittedTypeTarget {
    /// A type linked from a minimal group header with two aliases, with stored count two.
    ///
    /// The group header contains only a hard link and a reference-count message.
    AliasedGroup,
    /// A type with two direct aliases and stored count three.
    AliasedType,
    /// A named type with the supplied stored count, used by a dataset with a heap-backed attribute.
    Attributed(u32),
    /// A type referenced only by the dataset, without an explicit count record.
    Implicit,
    /// A named type with the supplied stored count.
    Named(u32),
}

impl CommittedTypeTarget {
    /// Writes a dataset using a committed datatype and returns the datatype's physical header
    /// address.
    ///
    /// The compact dataset `source` contains `[17.0]` using a little-endian 64-bit floating-point
    /// datatype. With [`Self::Attributed`], its attribute `a` contains the string `before`.
    ///
    /// # Panics
    ///
    /// Panics if fixture construction or file I/O fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure::File;
    /// use test_util_hdf5::__private::CommittedTypeTarget;
    ///
    /// let dir = tempfile::tempdir()?;
    /// let path = dir.path().join("committed-type.h5");
    /// let datatype_address = CommittedTypeTarget::Named(2).write(&path);
    /// assert!(datatype_address > 0);
    /// let file = File::open(&path)?;
    /// assert_eq!(file.root().named_datatypes()?, ["type"]);
    /// let dataset = file.dataset("source")?;
    /// assert_eq!(dataset.shape()?, [1]);
    /// assert_eq!(dataset.read_f64()?, [17.0]);
    /// drop(dataset);
    /// file.close()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn write(self, path: &Path) -> u64 {
        let description = self;
        let mut fixture = Fixture::new();
        let mut records = vec![Message::new(MessageType::DATATYPE, &datatype::f64_le())];
        records.extend(description.count().map(Message::reference_count));
        let target = fixture.append_header(records);
        match description {
            Self::Implicit => {}
            Self::AliasedGroup => {
                let group = fixture.append_header(vec![
                    global_heap_fixture::hard_link("type", target),
                    Message::reference_count(2),
                ]);
                fixture.link("group", group);
                fixture.link("alias", group);
            }
            Self::AliasedType | Self::Attributed(_) | Self::Named(_) => {
                fixture.link("type", target);
                if matches!(description, Self::AliasedType) {
                    fixture.link("alias", target);
                }
            }
        }
        let mut records = Form::CompactDataset.messages(
            &mut fixture,
            &Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
                DatatypeByteOrder::LittleEndian,
            )),
            &17_f64.to_le_bytes(),
        );
        let message = records
            .iter_mut()
            .find(|message| message.msg_type == MessageType::DATATYPE)
            .unwrap();
        *message = Message::new(
            MessageType::DATATYPE,
            &datatype::committed_reference(target),
        );
        message.flags = MessageFlags::SHARED;
        if matches!(description, Self::Attributed(_)) {
            let parent = fixture.collection(&[Object::new(7, b"before")]);
            records.extend(Form::CompactAttribute.messages(
                &mut fixture,
                &global_heap_fixture::vlen_string(),
                &global_heap_fixture::reference(6, parent, 7),
            ));
        }
        fixture.object("source", records);
        fixture.write(path);
        target
    }

    fn count(&self) -> Option<u32> {
        match self {
            Self::AliasedGroup => Some(2),
            Self::AliasedType => Some(3),
            Self::Implicit => None,
            Self::Attributed(count) | Self::Named(count) => Some(*count),
        }
    }
}

impl From<Option<u32>> for CommittedTypeTarget {
    /// Selects [`Self::Named`] for a count or [`Self::Implicit`] for [`None`].
    fn from(count: Option<u32>) -> Self {
        count.map_or(Self::Implicit, Self::Named)
    }
}
