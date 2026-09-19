//! Defines HDF5 object header message type identifiers.
//!
//! [`MessageType`] preserves every raw identifier and gives symbolic names to the message types
//! this crate understands. Object header message types are defined in "Data Object Header Messages"
//! of the [format specification, version 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_dataobject_hdr_msg

use core::fmt;

/// Identifies an HDF5 object header message type.
///
/// Every [`u16`] is a valid `MessageType`, so an identifier that this crate does not recognize
/// remains representable without losing its raw value. Recognized identifiers are exposed as
/// associated constants such as [`Self::DATASPACE`].
///
/// `MessageType` is carried by [`Error::MissingMessage`](crate::Error::MissingMessage). Its
/// transparent representation gives it the same size, alignment, and ABI as [`u16`].
///
/// The format specification calls the on-disk value the "Header Message Type" and defines its
/// values in "Data Object Header Messages" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_dataobject_hdr_msg
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MessageType(u16);

macro_rules! message_types {
    (
        $(
            $(#[$meta:meta])*
            $constant:ident / $compat:ident = $value:literal => $display:literal;
        )+
    ) => {
        impl MessageType {
            $(
                $(#[$meta])*
                pub const $constant: Self = Self($value);
            )+

            #[inline]
            const fn display_name(self) -> Option<&'static str> {
                match self {
                    $(
                        Self::$constant => Some($display),
                    )+
                    _ => None,
                }
            }

            /// Returns whether this crate recognizes the message type.
            ///
            /// Recognition means that the identifier has an associated constant and a symbolic
            /// name in this module. An identifier can be defined by the HDF5 specification without
            /// being recognized by this crate.
            #[inline]
            pub const fn is_known(self) -> bool {
                matches!(self, $(Self::$constant)|+)
            }
        }

        impl fmt::Debug for MessageType {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match *self {
                    $(
                        Self::$constant => f.write_str(stringify!($compat)),
                    )+
                    other => f.debug_tuple("Unknown").field(&other.0).finish(),
                }
            }
        }

        // Duplicate discriminants are rejected by the compiler, so this enum validates that every
        // recognized message in the registry has a distinct raw identifier.
        #[allow(dead_code)]
        #[repr(u16)]
        enum KnownMessageTypeDiscriminant {
            $(
                $compat = $value,
            )+
        }

        #[allow(non_upper_case_globals)]
        impl MessageType {
            $(
                #[doc(hidden)]
                $(#[$meta])*
                pub const $compat: Self = Self::$constant;
            )+
        }

        #[cfg(test)]
        const KNOWN: &[(u16, MessageType, &'static str, &'static str)] = &[
            $(
                (
                    $value,
                    MessageType::$constant,
                    $display,
                    stringify!($compat),
                ),
            )+
        ];
    };
}

message_types! {
    /// Identifies an object header message that the reader ignores.
    NIL / Nil = 0x0000 => "NIL";

    /// Identifies the dimensionality and extents of a dataset.
    DATASPACE / Dataspace = 0x0001 => "dataspace";

    /// Identifies the storage and indexing information for a group's links.
    LINK_INFO / LinkInfo = 0x0002 => "link info";

    /// Identifies the datatype of a dataset or committed datatype.
    DATATYPE / Datatype = 0x0003 => "datatype";

    /// Identifies a fill value encoded in the original fill-value message format.
    FILL_VALUE_OLD / FillValueOld = 0x0004 => "fill value (old)";

    /// Identifies the value used to fill unwritten dataset elements.
    FILL_VALUE / FillValue = 0x0005 => "fill value";

    /// Identifies a link stored in an object header.
    LINK / Link = 0x0006 => "link";

    /// Identifies dataset data stored in files outside the HDF5 file.
    ///
    /// The message lists the external files and the portions of those files used for the dataset.
    /// This crate does not follow external files and rejects a dataset that uses them.
    ///
    /// `H5Pset_external` configures this storage in the C library. The message is defined in
    /// "The Data Storage - External Data Files Message" of the
    /// [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_external
    EXTERNAL_DATA_FILES / ExternalDataFiles = 0x0007 => "external data files";

    /// Identifies how a dataset's raw data is stored.
    DATA_LAYOUT / DataLayout = 0x0008 => "data layout";

    // 0x0009 is the Bogus message. "The Bogus Message" of the format specification, version 4.0,
    // reserves it for testing and says that it should never be stored in a valid file.

    /// Identifies storage parameters for a group.
    GROUP_INFO / GroupInfo = 0x000A => "group info";

    /// Identifies the filter pipeline applied to a dataset's raw data.
    FILTER_PIPELINE / FilterPipeline = 0x000B => "filter pipeline";

    /// Identifies an attribute stored in an object header.
    ATTRIBUTE / Attribute = 0x000C => "attribute";

    // "Data Object Header Messages" of the format specification, version 4.0, defines 0x000D as
    // Object Comment and 0x000E as Object Modification Time (Old). This crate does not interpret
    // either message.

    /// Identifies the table used to locate shared object header messages.
    SHARED_MESSAGE_TABLE / SharedMessageTable = 0x000F => "shared message table";

    /// Identifies a block that contains additional object header messages.
    OBJECT_HEADER_CONTINUATION / ObjectHeaderContinuation =
        0x0010 => "object header continuation";

    /// Identifies the symbol-table storage used by an old-style group.
    SYMBOL_TABLE / SymbolTable = 0x0011 => "symbol table";

    /// Identifies the last modification time of an object.
    OBJECT_MODIFICATION_TIME / ObjectModificationTime =
        0x0012 => "object modification time";

    /// Identifies non-default B-tree K values stored with an object.
    B_TREE_K_VALUES / BTreeKValues = 0x0013 => "B-tree K values";

    // "The Driver Info Message" of the format specification, version 4.0, defines 0x0014. This
    // crate does not interpret that message.

    /// Identifies the storage and indexing information for densely stored attributes.
    ATTRIBUTE_INFO / AttributeInfo = 0x0015 => "attribute info";

    /// Identifies the number of hard links that point to an object.
    OBJECT_REFERENCE_COUNT / ObjectReferenceCount =
        0x0016 => "object reference count";

    /// Identifies the information used to manage free space in the file.
    FILE_SPACE_INFO / FileSpaceInfo = 0x0017 => "file space info";
}

impl MessageType {
    /// Constructs a message type without interpreting the raw identifier.
    ///
    /// Every [`u16`] is accepted. Use [`Self::is_known`] or [`Self::unknown_id`] when the
    /// distinction between recognized and unrecognized identifiers matters.
    #[inline]
    pub const fn from_u16(value: u16) -> Self {
        Self(value)
    }

    /// Returns the raw HDF5 message type identifier.
    #[inline]
    pub const fn to_u16(self) -> u16 {
        self.0
    }

    /// Returns whether this crate does not recognize the message type.
    ///
    /// An unrecognized identifier can still be defined by the HDF5 specification. It has no
    /// associated symbolic name in this crate.
    #[inline]
    pub const fn is_unknown(self) -> bool {
        !self.is_known()
    }

    /// Returns the raw identifier when this crate does not recognize the message type.
    ///
    /// Returns [`None`] when [`Self::is_known`] is `true`.
    #[inline]
    pub const fn unknown_id(self) -> Option<u16> {
        if self.is_known() { None } else { Some(self.0) }
    }

    /// Constructs a message type without interpreting the raw identifier.
    ///
    /// The value can identify either a recognized or an unrecognized message. This function is
    /// equivalent to [`Self::from_u16`].
    #[doc(hidden)]
    #[allow(non_snake_case)]
    #[inline]
    pub const fn Unknown(value: u16) -> Self {
        Self(value)
    }
}

impl From<u16> for MessageType {
    #[inline]
    fn from(value: u16) -> Self {
        Self(value)
    }
}

impl From<MessageType> for u16 {
    #[inline]
    fn from(value: MessageType) -> Self {
        value.0
    }
}

impl fmt::Display for MessageType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.display_name() {
            Some(name) => f.write_str(name),
            None => write!(f, "unknown message 0x{:04x}", self.0),
        }
    }
}

// `repr(transparent)` guarantees these properties. The assertions make the representation
// requirement explicit at compile time.
const _: () = {
    assert!(core::mem::size_of::<MessageType>() == core::mem::size_of::<u16>());
    assert!(core::mem::align_of::<MessageType>() == core::mem::align_of::<u16>());
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representation_matches_u16() {
        assert_eq!(
            core::mem::size_of::<MessageType>(),
            core::mem::size_of::<u16>()
        );
        assert_eq!(
            core::mem::align_of::<MessageType>(),
            core::mem::align_of::<u16>()
        );
    }

    #[test]
    fn every_raw_identifier_roundtrips() {
        for raw in u16::MIN..=u16::MAX {
            let message = MessageType::from_u16(raw);

            assert_eq!(message.to_u16(), raw);
            assert_eq!(u16::from(MessageType::from(raw)), raw);
        }
    }

    #[test]
    fn known_identifiers_are_strictly_increasing() {
        for pair in KNOWN.windows(2) {
            let previous = pair[0].0;
            let next = pair[1].0;

            assert!(previous < next, "{previous:#06x} must precede {next:#06x}");
        }
    }

    #[test]
    fn known_types_roundtrip() {
        for &(raw, expected, display, _) in KNOWN {
            let message = MessageType::from_u16(raw);

            assert_eq!(message, expected);
            assert_eq!(message.to_u16(), raw);
            assert!(message.is_known());
            assert!(!message.is_unknown());
            assert_eq!(message.unknown_id(), None);
            assert_eq!(message.display_name(), Some(display));
        }
    }

    #[test]
    fn unknown_type_preserves_its_identifier() {
        let message = MessageType::from_u16(0x00FF);

        assert_eq!(message.to_u16(), 0x00FF);
        assert!(!message.is_known());
        assert!(message.is_unknown());
        assert_eq!(message.unknown_id(), Some(0x00FF));
        assert_eq!(message.display_name(), None);
    }

    #[test]
    fn bogus_message_is_not_recognized() {
        let message = MessageType::from_u16(0x0009);

        assert!(message.is_unknown());
        assert_eq!(message.unknown_id(), Some(0x0009));
    }

    #[test]
    fn defined_but_unsupported_types_are_not_recognized() {
        for raw in [0x000D, 0x000E, 0x0014] {
            let message = MessageType::from_u16(raw);

            assert!(message.is_unknown(), "{raw:#06x}");
            assert_eq!(message.unknown_id(), Some(raw));
            assert_eq!(message.to_u16(), raw);
        }
    }

    #[test]
    fn external_data_files_is_recognized() {
        let message = MessageType::from_u16(0x0007);

        assert_eq!(message, MessageType::EXTERNAL_DATA_FILES);
        assert_eq!(message.to_u16(), 0x0007);
        assert_eq!(message.display_name(), Some("external data files"));
    }

    #[test]
    fn from_conversions_are_lossless() {
        let message = MessageType::from(0x1234);

        assert_eq!(message.to_u16(), 0x1234);
        assert_eq!(u16::from(message), 0x1234);
    }

    #[test]
    fn camel_case_constants_work_as_patterns() {
        let message = MessageType::from_u16(0x0001);

        assert!(matches!(message, MessageType::Dataspace));
    }

    #[test]
    fn unknown_constructor_preserves_the_identifier() {
        assert_eq!(MessageType::Unknown(0x00FF), MessageType::from_u16(0x00FF));
    }
}

#[cfg(all(test, feature = "std"))]
mod display_tests {
    use super::*;

    #[test]
    fn every_known_message_uses_its_format_name() {
        for &(raw, message, display, _) in KNOWN {
            assert_eq!(message.to_string(), display, "{raw:#06x}");
        }
    }

    #[test]
    fn an_unknown_message_reports_its_raw_identifier() {
        assert_eq!(
            MessageType::from_u16(0x00FF).to_string(),
            "unknown message 0x00ff"
        );
    }

    #[test]
    fn every_known_message_uses_its_symbolic_debug_name() {
        for &(raw, message, _, debug) in KNOWN {
            assert_eq!(format!("{message:?}"), debug, "{raw:#06x}");
        }
    }

    #[test]
    fn an_unknown_message_uses_the_unknown_debug_form() {
        assert_eq!(
            format!("{:?}", MessageType::from_u16(0x00FF)),
            "Unknown(255)"
        );
    }
}
