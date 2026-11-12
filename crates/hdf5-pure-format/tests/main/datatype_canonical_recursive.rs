use hdf5_pure_core::{Datatype, FormatError, ReferenceType, VariableLengthType};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::{DatatypeMessage, OffsetWidth};
use rstest::rstest;
use test_util::datatype;
use test_util::datatype::Class;
use test_util::datatype::containers::{self, CompoundField, EnumValue, MemberOffset};
use test_util::datatype::leaf::{self, Leaf, OpaqueTag, Order, VaxVersion, Version};

#[rstest]
#[case::integer(Parent::Atom(AtomicParent::Integer))]
#[case::float(Parent::Atom(AtomicParent::Float))]
#[case::time(Parent::Atom(AtomicParent::Time))]
#[case::string(Parent::Atom(AtomicParent::String))]
#[case::bitfield(Parent::Atom(AtomicParent::Bitfield))]
#[case::opaque(Parent::Atom(AtomicParent::Opaque))]
#[case::vax(Parent::Vax)]
#[case::compound(Parent::Compound)]
#[case::enumeration(Parent::Enumeration)]
#[case::sequence(Parent::Sequence)]
#[case::variable_string(Parent::String)]
#[case::complex(Parent::Complex(false))]
#[case::vax_complex(Parent::Complex(true))]
fn datatype_canonical_recursive_preserves_properties_and_minimal_versions(
    #[case] parent: Parent,
    #[values(Wrapper::Sequence, Wrapper::Array)] wrapper: Wrapper,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let source = parent.bytes(OffsetWidth::Eight, true);
    let semantic = DatatypeMessage::parse(&source)
        .unwrap()
        .0
        .to_datatype()
        .unwrap();
    let expected = wrapper.bytes(
        width,
        &parent.bytes(width, false),
        parent.size(width),
        parent.version(false),
    );
    assert_eq!(
        format::serialize_datatype_with_offset_width(&wrapper.wrap(semantic), width).unwrap(),
        expected
    );
}

#[rstest]
#[case::object(ReferenceType::Object, [0; 3], 8)]
#[case::region(ReferenceType::DatasetRegion, REGION_REFERENCE, 12)]
fn datatype_canonical_recursive_preserves_legacy_reference_parents(
    #[case] ref_type: ReferenceType,
    #[case] flags: [u8; 3],
    #[case] size: u32,
    #[values(Wrapper::Sequence, Wrapper::Array)] wrapper: Wrapper,
    #[values(OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight)] width: OffsetWidth,
) {
    let parent = Datatype::Reference { size, ref_type };
    let bytes = datatype::header(Class::REFERENCE, Version::One.get(), flags, size);
    assert_eq!(
        format::serialize_datatype_with_offset_width(&wrapper.wrap(parent), width).unwrap(),
        wrapper.bytes(width, &bytes, size, Version::One)
    );
}

#[rstest]
#[case::compound(EmptyModel::Compound)]
#[case::enumeration(EmptyModel::Enumeration)]
#[case::zero_array(EmptyModel::Array)]
fn datatype_canonical_recursive_preserves_constructed_models_outside_checked_messages(
    #[case] model: EmptyModel,
    #[values(Wrapper::Sequence, Wrapper::Array)] wrapper: Wrapper,
) {
    let parent = model.datatype();
    let expected = wrapper.bytes(
        OffsetWidth::Eight,
        &model.bytes(),
        model.size(),
        Version::Three,
    );
    assert_eq!(
        format::serialize_datatype(&wrapper.wrap(parent)).unwrap(),
        expected
    );
    assert_eq!(
        DatatypeMessage::parse(&expected).unwrap_err(),
        model.error()
    );
}

#[derive(Clone, Copy, Debug)]
enum Wrapper {
    Array,
    Sequence,
}

impl Wrapper {
    fn wrap(self, parent: Datatype) -> Datatype {
        match self {
            Self::Array => Datatype::Array {
                base_type: Box::new(parent),
                dimensions: vec![2, 3],
            },
            Self::Sequence => Datatype::VariableLength(VariableLengthType::sequence(parent)),
        }
    }

    fn bytes(self, width: OffsetWidth, parent: &[u8], size: u32, version: Version) -> Vec<u8> {
        match self {
            Self::Array => datatype::array(
                version.get().max(Version::Three.get()),
                [0; 3],
                size * 6,
                2,
                &[2, 3],
                parent,
            ),
            Self::Sequence => {
                datatype::variable_length(version.get(), [0; 3], descriptor_size(width), parent)
            }
        }
    }
}

#[derive(Debug)]
enum Parent {
    Atom(AtomicParent),
    Complex(bool),
    Compound,
    Enumeration,
    Sequence,
    String,
    Vax,
}

impl Parent {
    fn bytes(&self, width: OffsetWidth, source: bool) -> Vec<u8> {
        let version = self.version(source);
        let child_version = if source { Version::Five } else { Version::One };
        let integer = Leaf::Integer.bytes(child_version);
        match self {
            Self::Atom(leaf) => leaf.leaf().bytes(version),
            Self::Complex(vax) => {
                let parent = if *vax {
                    leaf::vax_bytes(if source {
                        VaxVersion::Five
                    } else {
                        VaxVersion::Three
                    })
                } else {
                    Leaf::Float(Order::Big).bytes(child_version)
                };
                datatype::complex(version.get(), HOMOGENEOUS_COMPLEX, 16, &parent)
            }
            Self::Compound => containers::compound(
                version.get(),
                [1, 0, 0],
                8,
                MemberOffset::One,
                &[CompoundField {
                    name: b"field",
                    offset: 0,
                    datatype: &integer,
                }],
            ),
            Self::Enumeration => containers::enumeration(
                version.get(),
                [2, 0, 0],
                4,
                &integer,
                &[
                    EnumValue {
                        name: b"first",
                        value: &[0; 4],
                    },
                    EnumValue {
                        name: b"second",
                        value: &[1; 4],
                    },
                ],
            ),
            Self::Sequence => {
                datatype::variable_length(version.get(), [0; 3], descriptor_size(width), &integer)
            }
            Self::String => datatype::variable_length(
                version.get(),
                UTF8_SPACE_PADDED,
                descriptor_size(width),
                &integer,
            ),
            Self::Vax => leaf::vax_bytes(if source {
                VaxVersion::Five
            } else {
                VaxVersion::Three
            }),
        }
    }

    fn size(&self, width: OffsetWidth) -> u32 {
        match self {
            Self::Atom(leaf) => leaf.size(),
            Self::Complex(_) => 16,
            Self::Compound | Self::Vax => 8,
            Self::Enumeration => 4,
            Self::Sequence | Self::String => descriptor_size(width),
        }
    }

    fn version(&self, source: bool) -> Version {
        if source {
            return Version::Five;
        }
        match self {
            Self::Complex(_) => Version::Five,
            Self::Compound | Self::Enumeration | Self::Vax => Version::Three,
            Self::Atom(_) | Self::Sequence | Self::String => Version::One,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum AtomicParent {
    Bitfield,
    Float,
    Integer,
    Opaque,
    String,
    Time,
}

impl AtomicParent {
    fn leaf(self) -> Leaf {
        match self {
            Self::Bitfield => Leaf::Bitfield,
            Self::Float => Leaf::Float(Order::Big),
            Self::Integer => Leaf::Integer,
            Self::Opaque => Leaf::Opaque(OpaqueTag::Aligned),
            Self::String => Leaf::string(),
            Self::Time => Leaf::Time,
        }
    }

    fn size(self) -> u32 {
        match self {
            Self::Bitfield | Self::Integer | Self::Time => 4,
            Self::Float => 8,
            Self::Opaque => 3,
            Self::String => 7,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EmptyModel {
    Array,
    Compound,
    Enumeration,
}

impl EmptyModel {
    fn datatype(self) -> Datatype {
        let parent = DatatypeMessage::parse(&Leaf::Integer.bytes(Version::One))
            .unwrap()
            .0
            .to_datatype()
            .unwrap();
        match self {
            Self::Array => Datatype::Array {
                base_type: Box::new(parent),
                dimensions: vec![0],
            },
            Self::Compound => Datatype::Compound {
                size: self.size(),
                members: vec![],
            },
            Self::Enumeration => Datatype::Enumeration {
                size: self.size(),
                base_type: Box::new(parent),
                members: vec![],
            },
        }
    }

    fn bytes(self) -> Vec<u8> {
        let integer = Leaf::Integer.bytes(Version::One);
        match self {
            Self::Array => datatype::array(Version::Three.get(), [0; 3], 0, 1, &[0], &integer),
            Self::Compound => {
                datatype::header(Class::COMPOUND, Version::Three.get(), [0; 3], self.size())
            }
            Self::Enumeration => {
                containers::enumeration(Version::Three.get(), [0; 3], self.size(), &integer, &[])
            }
        }
    }

    fn size(self) -> u32 {
        match self {
            Self::Array => 0,
            Self::Compound | Self::Enumeration => 4,
        }
    }
    fn error(self) -> FormatError {
        match self {
            Self::Array => FormatError::ZeroSizedDatatype {
                class: Class::ARRAY.0,
            },
            Self::Compound => FormatError::InvalidCompoundMemberCount { count: 0 },
            Self::Enumeration => FormatError::InvalidEnumerationMemberCount { count: 0 },
        }
    }
}

fn descriptor_size(width: OffsetWidth) -> u32 {
    HEAP_REFERENCE_OVERHEAD + u32::from(width.get())
}

/// The eight descriptor bytes outside the heap address.
///
/// `H5T__vlen_disk_write` in `H5Tvlen.c`, HDF5 2.2.0, writes the four-byte sequence length.
/// The four-byte index is defined in "Global Heap" of the [format specification,
/// version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_globalheap
const HEAP_REFERENCE_OVERHEAD: u32 = 8;
/// The flags for a version 5 homogeneous rectangular complex datatype message.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const HOMOGENEOUS_COMPLEX: [u8; 3] = [1, 0, 0];
/// The dataset-region reference flags for datatype message versions 1 through 3.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const REGION_REFERENCE: [u8; 3] = [1, 0, 0];
/// The space-padded UTF-8 variable-length string flags for datatype message versions 1 through 5.
///
/// Defined in "The Datatype Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
const UTF8_SPACE_PADDED: [u8; 3] = [0x21, 1, 0];
