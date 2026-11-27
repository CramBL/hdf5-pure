use super::view::LinkTargetView;
use crate::datatype::CharacterSet;
use crate::width::UintWidth;

/// The checked fields of an owned link, borrowed for encoding.
pub(crate) trait StoredLink: sealed::Sealed {
    /// Returns the name bytes and the width of their length field.
    fn name_field(&self) -> (&[u8], UintWidth);
    /// Returns the target and its encoded value length, or zero for a hard link.
    fn target_field(&self) -> (LinkTargetView<'_>, u16);
    fn creation_index(&self) -> Option<u64>;
    fn charset_field(&self) -> Option<CharacterSet>;
}

mod sealed {
    pub(crate) trait Sealed {}

    impl Sealed for super::super::wire::LinkMessageV1 {}
    impl Sealed for crate::link::storage::OwnedLink {}
}
