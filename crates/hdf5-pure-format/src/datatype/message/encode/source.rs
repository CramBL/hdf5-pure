//! Shared property views over semantic datatypes and retained descriptions.

use hdf5_pure_core::__private;
use hdf5_pure_core::{
    CompoundMember, Datatype, DatatypeMemberName, FloatingPointType, ReferenceType,
    VariableLengthType,
};

use super::container::ContainerProperties;
use crate::datatype::legacy_member::CompoundV1Shape;
use crate::datatype::message::atomic::AtomicProperties;
use crate::datatype::message::compound_layout::CompoundFields;
use crate::datatype::message::name_layout::NameLayout;
use crate::datatype::message::representation;
use crate::datatype::message::{
    CompoundV1Member, DatatypeMessage, DatatypeMessageMember, DatatypeProperties, DatatypeVersion,
};
use crate::datatype::references::{OBJECT_REFERENCE_KIND, REGION_REFERENCE_KIND};

/// The source of a datatype description and its message version policy.
#[derive(Debug)]
pub(super) enum Description<'a> {
    /// A semantic datatype that uses canonical versions.
    Canonical(&'a Datatype),
    /// The floating-point component of a semantic complex datatype.
    FloatingPoint(FloatingPointType),
    /// A checked description that retains its own message version.
    Preserved(&'a DatatypeMessage),
}

impl Description<'_> {
    /// Returns the retained version, or none when the writer chooses a canonical version.
    pub(super) fn version(&self) -> Option<DatatypeVersion> {
        match self {
            Self::Preserved(message) => Some(message.version()),
            _ => None,
        }
    }

    /// Returns the borrowed property fields used for sizing and writing.
    ///
    /// For an unsupported semantic description, returns [`Body::Unsupported`] with the reason.
    pub(super) fn body(&self) -> Body<'_> {
        match self {
            Self::Canonical(datatype) => Body::canonical(datatype),
            Self::FloatingPoint(datatype) => {
                Body::Atomic(AtomicProperties::FloatingPoint(datatype))
            }
            Self::Preserved(message) => Body::preserved(message),
        }
    }
}

/// The fields needed to size and write one description.
pub(super) enum Body<'a> {
    Atomic(AtomicProperties<'a>),
    Container(ContainerProperties<'a>),
    Reference {
        size: u32,
        kind: u8,
    },
    /// The reason preparation cannot encode this semantic datatype.
    Unsupported(&'static str),
}

impl<'a> Body<'a> {
    fn canonical(datatype: &'a Datatype) -> Self {
        let properties = match datatype {
            Datatype::FixedPoint(datatype) => {
                return Self::Atomic(AtomicProperties::FixedPoint(datatype));
            }
            Datatype::FloatingPoint(datatype) => {
                return Self::Atomic(AtomicProperties::FloatingPoint(datatype));
            }
            Datatype::Time(datatype) => return Self::Atomic(AtomicProperties::Time(datatype)),
            Datatype::String {
                size,
                padding,
                charset,
            } => {
                return Self::Atomic(AtomicProperties::String {
                    size: *size,
                    padding,
                    charset,
                });
            }
            Datatype::BitField(datatype) => {
                return Self::Atomic(AtomicProperties::BitField(datatype));
            }
            Datatype::Opaque { size, tag } => {
                return Self::Atomic(AtomicProperties::Opaque { size: *size, tag });
            }
            Datatype::Reference { size, ref_type } => {
                let kind = match ref_type {
                    ReferenceType::Object => OBJECT_REFERENCE_KIND,
                    ReferenceType::DatasetRegion => REGION_REFERENCE_KIND,
                    _ => {
                        return Self::Unsupported(
                            "a semantic reference kind has no legacy encoding",
                        );
                    }
                };
                return Self::Reference { size: *size, kind };
            }
            Datatype::Compound { size, members } => ContainerProperties::Compound {
                size: *size,
                members: CompoundMembers::Canonical(members),
                fields: CompoundFields::for_version(DatatypeVersion::Three, *size),
            },
            Datatype::Enumeration(datatype) => ContainerProperties::Enumeration {
                size: datatype.size().get(),
                base_type: Description::Canonical(datatype.base_type()),
                members: datatype.members(),
                name_layout: NameLayout::Unpadded,
                offsets: __private::enumeration_base_offsets(datatype),
            },
            Datatype::VariableLength(kind) => ContainerProperties::VariableLength {
                base_type: Description::Canonical(kind.base_type()),
                string: match kind {
                    VariableLengthType::Sequence { .. } => None,
                    VariableLengthType::String(string) => {
                        Some((string.padding(), string.charset()))
                    }
                },
            },
            Datatype::Array {
                base_type,
                dimensions,
            } => ContainerProperties::Array {
                base_type: Description::Canonical(base_type),
                dimensions,
            },
            Datatype::Complex(complex) => ContainerProperties::Complex {
                component: Description::FloatingPoint(complex.component()),
                size: Some(complex.size().get()),
            },
            _ => return Self::Unsupported("a semantic datatype has no canonical encoding"),
        };
        Self::Container(properties)
    }

    fn preserved(message: &'a DatatypeMessage) -> Self {
        let properties = match message.properties() {
            DatatypeProperties::FixedPoint(datatype) => {
                return Self::Atomic(AtomicProperties::FixedPoint(datatype));
            }
            DatatypeProperties::FloatingPoint(datatype) => {
                return Self::Atomic(AtomicProperties::FloatingPoint(datatype));
            }
            DatatypeProperties::Time(datatype) => {
                return Self::Atomic(AtomicProperties::Time(datatype));
            }
            DatatypeProperties::String {
                size,
                padding,
                charset,
            } => {
                return Self::Atomic(AtomicProperties::String {
                    size: *size,
                    padding,
                    charset,
                });
            }
            DatatypeProperties::BitField(datatype) => {
                return Self::Atomic(AtomicProperties::BitField(datatype));
            }
            DatatypeProperties::Opaque { size, tag } => {
                return Self::Atomic(AtomicProperties::Opaque { size: *size, tag });
            }
            DatatypeProperties::Compound { size, members } => ContainerProperties::Compound {
                size: *size,
                members: CompoundMembers::Preserved(members),
                fields: CompoundFields::for_version(message.version(), *size),
            },
            DatatypeProperties::CompoundV1 { size, members } => ContainerProperties::Compound {
                size: *size,
                members: CompoundMembers::PreservedV1(members),
                fields: CompoundFields::LegacyWithShape,
            },
            DatatypeProperties::Enumeration { base_type, members } => {
                ContainerProperties::Enumeration {
                    size: representation::enumeration_value_width(members),
                    base_type: Description::Preserved(base_type),
                    members,
                    name_layout: NameLayout::for_version(message.version()),
                    offsets: message.enumeration_base_offsets(),
                }
            }
            DatatypeProperties::VariableLengthSequence { base_type } => {
                ContainerProperties::VariableLength {
                    base_type: Description::Preserved(base_type),
                    string: None,
                }
            }
            DatatypeProperties::VariableLengthString {
                padding,
                charset,
                base_type,
            } => ContainerProperties::VariableLength {
                base_type: Description::Preserved(base_type),
                string: Some((padding, charset)),
            },
            DatatypeProperties::Array {
                base_type,
                dimensions,
            } => ContainerProperties::Array {
                base_type: Description::Preserved(base_type),
                dimensions,
            },
            DatatypeProperties::Complex { component } => ContainerProperties::Complex {
                component: Description::Preserved(component),
                size: None,
            },
        };
        Self::Container(properties)
    }
}

/// The member list, including inline array shapes for retained version 1 descriptions.
pub(super) enum CompoundMembers<'a> {
    Canonical(&'a [CompoundMember]),
    Preserved(&'a [DatatypeMessageMember]),
    PreservedV1(&'a [CompoundV1Member]),
}

impl CompoundMembers<'_> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Canonical(members) => members.len(),
            Self::Preserved(members) => members.len(),
            Self::PreservedV1(members) => members.len(),
        }
    }

    /// Borrows a member's encoding fields under the version policy of its compound.
    ///
    /// # Panics
    ///
    /// Panics if `index` is outside the member list.
    pub(super) fn member(&self, index: usize) -> CompoundMemberSource<'_> {
        match self {
            Self::Canonical(members) => {
                let (name, offset, datatype) = __private::compound_member_parts(&members[index]);
                CompoundMemberSource {
                    name,
                    offset,
                    child: Description::Canonical(datatype),
                    shape: None,
                }
            }
            Self::Preserved(members) => CompoundMemberSource::preserved(&members[index], None),
            Self::PreservedV1(members) => {
                let member = &members[index];
                CompoundMemberSource::preserved(member.member(), Some(member.shape()))
            }
        }
    }

    pub(super) fn member_name(&self, index: usize) -> Option<&DatatypeMemberName> {
        match self {
            Self::Canonical(members) => members.get(index).map(|member| &member.name),
            Self::Preserved(members) => members.get(index).map(DatatypeMessageMember::name),
            Self::PreservedV1(members) => members.get(index).map(|member| member.member().name()),
        }
    }
}

pub(super) struct CompoundMemberSource<'a> {
    pub(super) name: &'a DatatypeMemberName,
    pub(super) offset: u64,
    pub(super) child: Description<'a>,
    /// The version 1 dimension block, including the rank-zero block for a scalar.
    pub(super) shape: Option<CompoundV1Shape>,
}

impl<'a> CompoundMemberSource<'a> {
    fn preserved(member: &'a DatatypeMessageMember, shape: Option<CompoundV1Shape>) -> Self {
        Self {
            name: member.name(),
            offset: member.byte_offset(),
            child: Description::Preserved(member.datatype()),
            shape,
        }
    }
}
