//! The flags of an object header message record.

use core::fmt;
use core::ops::BitOr;

use crate::access_mode::AccessMode;

/// The flags of an object header message record, its "Header Message #n Flags" field.
///
/// The format defines all eight bits of the field, so every byte a record stores is a combination
/// of the eight flags. A version 2 object header stores the same field as a version 1 header,
/// defined in "Version 1 Data Object Header Prefix" of the [format specification, version
/// 4.0][spec]. The C library defines the same eight bits as `H5O_MSG_FLAG_CONSTANT` through
/// `H5O_MSG_FLAG_FAIL_IF_UNKNOWN_ALWAYS` in `H5Oprivate.h`, release 2.2.0.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct MessageFlags(u8);

impl MessageFlags {
    /// Wraps the flags byte a parser read from a message record.
    pub(crate) const fn new(byte: u8) -> Self {
        Self(byte)
    }

    /// Returns the byte a record stores for these flags.
    pub(crate) const fn get(self) -> u8 {
        self.0
    }

    /// Returns `true` if [`MessageFlags::CONSTANT`] is set.
    pub(crate) const fn is_constant(self) -> bool {
        self.contains(Self::CONSTANT)
    }

    /// Returns `true` if [`MessageFlags::SHARED`] is set.
    pub(crate) const fn is_shared(self) -> bool {
        self.contains(Self::SHARED)
    }

    /// Returns `true` if [`MessageFlags::FORBID_SHARING`] is set.
    pub(crate) const fn forbids_sharing(self) -> bool {
        self.contains(Self::FORBID_SHARING)
    }

    /// Returns `true` if [`MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE`] is set.
    pub(crate) const fn fails_if_unknown_and_open_for_write(self) -> bool {
        self.contains(Self::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE)
    }

    /// Returns `true` if [`MessageFlags::MARK_IF_UNKNOWN`] is set.
    pub(crate) const fn marks_if_unknown(self) -> bool {
        self.contains(Self::MARK_IF_UNKNOWN)
    }

    /// Returns `true` if [`MessageFlags::WAS_UNKNOWN`] is set.
    pub(crate) const fn was_unknown(self) -> bool {
        self.contains(Self::WAS_UNKNOWN)
    }

    /// Returns `true` if [`MessageFlags::SHAREABLE`] is set.
    pub(crate) const fn is_shareable(self) -> bool {
        self.contains(Self::SHAREABLE)
    }

    /// Returns `true` if [`MessageFlags::FAIL_IF_UNKNOWN_ALWAYS`] is set.
    pub(crate) const fn fails_if_unknown_always(self) -> bool {
        self.contains(Self::FAIL_IF_UNKNOWN_ALWAYS)
    }

    /// Returns `true` if a decoder under `access_mode` that cannot name the message's type must
    /// reject the object: [`MessageFlags::FAIL_IF_UNKNOWN_ALWAYS`] under either mode, and
    /// [`MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE`] under [`AccessMode::ReadWrite`] alone.
    pub(crate) const fn must_be_understood(self, access_mode: AccessMode) -> bool {
        self.fails_if_unknown_always()
            || (self.fails_if_unknown_and_open_for_write() && access_mode.is_read_write())
    }

    /// Returns `true` if no flag is set.
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the flags set in `self` and not in `other`.
    pub(crate) const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Returns `true` if every flag set in `other` is set in `self`.
    const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// No flag set.
    pub(crate) const NONE: Self = Self(0x00);
    /// The message data is constant, as a dataset's datatype message is.
    pub(crate) const CONSTANT: Self = Self(0x01);
    /// The message is stored outside the object header, and the record's body is the reference
    /// to it.
    pub(crate) const SHARED: Self = Self(0x02);
    /// The message should not be moved into shared storage.
    pub(crate) const FORBID_SHARING: Self = Self(0x04);
    /// A reader that does not know the message type should reject the object while the file is
    /// open for writing.
    pub(crate) const FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE: Self = Self(0x08);
    /// A reader that does not know the message type should set
    /// [`MessageFlags::WAS_UNKNOWN`] when it modifies the object.
    pub(crate) const MARK_IF_UNKNOWN: Self = Self(0x10);
    /// A library that did not know this message modified the object.
    pub(crate) const WAS_UNKNOWN: Self = Self(0x20);
    /// The message may be moved into shared storage.
    pub(crate) const SHAREABLE: Self = Self(0x40);
    /// A reader that does not know the message type should reject the object, whether the file is
    /// open for reading or for writing.
    pub(crate) const FAIL_IF_UNKNOWN_ALWAYS: Self = Self(0x80);
}

/// Combines two sets of flags.
impl BitOr for MessageFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Lists the flags that are set, as `MessageFlags(SHARED | FORBID_SHARING)`.
impl fmt::Debug for MessageFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let named = [
            (self.is_constant(), "CONSTANT"),
            (self.is_shared(), "SHARED"),
            (self.forbids_sharing(), "FORBID_SHARING"),
            (
                self.fails_if_unknown_and_open_for_write(),
                "FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE",
            ),
            (self.marks_if_unknown(), "MARK_IF_UNKNOWN"),
            (self.was_unknown(), "WAS_UNKNOWN"),
            (self.is_shareable(), "SHAREABLE"),
            (self.fails_if_unknown_always(), "FAIL_IF_UNKNOWN_ALWAYS"),
        ];
        f.write_str("MessageFlags(")?;
        let mut separator = "";
        for name in named.iter().filter_map(|&(set, name)| set.then_some(name)) {
            f.write_str(separator)?;
            f.write_str(name)?;
            separator = " | ";
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bit_of_every_flag_is_the_one_the_specification_gives() {
        assert_eq!(MessageFlags::NONE.get(), 0x00);
        assert_eq!(MessageFlags::CONSTANT.get(), 0x01);
        assert_eq!(MessageFlags::SHARED.get(), 0x02);
        assert_eq!(MessageFlags::FORBID_SHARING.get(), 0x04);
        assert_eq!(MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE.get(), 0x08);
        assert_eq!(MessageFlags::MARK_IF_UNKNOWN.get(), 0x10);
        assert_eq!(MessageFlags::WAS_UNKNOWN.get(), 0x20);
        assert_eq!(MessageFlags::SHAREABLE.get(), 0x40);
        assert_eq!(MessageFlags::FAIL_IF_UNKNOWN_ALWAYS.get(), 0x80);
    }

    #[test]
    fn a_flags_byte_survives_the_round_trip() {
        assert_eq!(MessageFlags::new(0x93).get(), 0x93);
    }

    #[test]
    fn an_empty_flags_byte_sets_no_flag() {
        assert!(MessageFlags::NONE.is_empty());
        assert_eq!(flags_set_in(MessageFlags::NONE), Vec::new());
    }

    #[test]
    fn each_bit_sets_the_one_flag_it_stands_for() {
        for bit in BITS {
            assert_eq!(flags_set_in(bit), vec![bit]);
        }
    }

    #[test]
    fn a_byte_with_two_bits_sets_both_of_their_flags() {
        let flags = MessageFlags::CONSTANT | MessageFlags::FORBID_SHARING;
        assert!(!flags.is_empty());
        assert!(flags.is_constant());
        assert!(flags.forbids_sharing());
        assert!(!flags.is_shared());
    }

    #[test]
    fn every_bit_of_a_byte_stands_for_a_flag() {
        let all = BITS
            .into_iter()
            .fold(MessageFlags::NONE, |all, bit| all | bit);
        assert_eq!(all.get(), u8::MAX);
        assert_eq!(flags_set_in(all), BITS.to_vec());
    }

    #[test]
    fn a_difference_clears_the_flags_it_names_and_keeps_the_rest() {
        let flags = MessageFlags::new(u8::MAX).difference(MessageFlags::SHARED);
        assert_eq!(flags.get(), 0xFD);
        assert!(
            MessageFlags::SHARED
                .difference(MessageFlags::SHARED)
                .is_empty()
        );
    }

    #[test]
    fn debug_names_every_flag_a_byte_sets() {
        assert_eq!(
            format!("{:?}", MessageFlags::new(0x06)),
            "MessageFlags(SHARED | FORBID_SHARING)"
        );
    }

    #[test]
    fn debug_names_no_flag_for_an_empty_flags_byte() {
        assert_eq!(format!("{:?}", MessageFlags::NONE), "MessageFlags()");
    }

    #[test]
    fn only_write_access_must_understand_a_message_flagged_fail_if_unknown_and_open_for_write() {
        let flags = MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE;
        assert!(!flags.must_be_understood(AccessMode::ReadOnly));
        assert!(flags.must_be_understood(AccessMode::ReadWrite));
    }

    #[test]
    fn either_access_must_understand_a_message_flagged_fail_if_unknown_always() {
        let flags = MessageFlags::FAIL_IF_UNKNOWN_ALWAYS;
        assert!(flags.must_be_understood(AccessMode::ReadOnly));
        assert!(flags.must_be_understood(AccessMode::ReadWrite));
    }

    #[test]
    fn either_access_reads_past_a_message_flagged_with_no_fail_if_unknown_bit() {
        let flags = MessageFlags::CONSTANT | MessageFlags::SHAREABLE;
        assert!(!flags.must_be_understood(AccessMode::ReadOnly));
        assert!(!flags.must_be_understood(AccessMode::ReadWrite));
    }

    fn flags_set_in(flags: MessageFlags) -> Vec<MessageFlags> {
        [
            (MessageFlags::CONSTANT, flags.is_constant()),
            (MessageFlags::SHARED, flags.is_shared()),
            (MessageFlags::FORBID_SHARING, flags.forbids_sharing()),
            (
                MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
                flags.fails_if_unknown_and_open_for_write(),
            ),
            (MessageFlags::MARK_IF_UNKNOWN, flags.marks_if_unknown()),
            (MessageFlags::WAS_UNKNOWN, flags.was_unknown()),
            (MessageFlags::SHAREABLE, flags.is_shareable()),
            (
                MessageFlags::FAIL_IF_UNKNOWN_ALWAYS,
                flags.fails_if_unknown_always(),
            ),
        ]
        .into_iter()
        .filter_map(|(bit, set)| set.then_some(bit))
        .collect()
    }

    const BITS: [MessageFlags; 8] = [
        MessageFlags::CONSTANT,
        MessageFlags::SHARED,
        MessageFlags::FORBID_SHARING,
        MessageFlags::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
        MessageFlags::MARK_IF_UNKNOWN,
        MessageFlags::WAS_UNKNOWN,
        MessageFlags::SHAREABLE,
        MessageFlags::FAIL_IF_UNKNOWN_ALWAYS,
    ];
}
