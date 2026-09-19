//! Object header message record flags.

use core::fmt;
use core::ops::BitOr;

use crate::access_mode::AccessMode;

/// The flags of an object header message record, its "Header Message #n Flags" field.
///
/// The format defines all eight bits of the field, so every byte is a valid combination of flags.
/// Version 2 object headers store the same field as version 1 object headers, defined in "Version
/// 1 Data Object Header Prefix" of the [format specification, version 4.0][spec].
///
/// The C library defines the same eight bits as `H5O_MSG_FLAG_CONSTANT` through
/// `H5O_MSG_FLAG_FAIL_IF_UNKNOWN_ALWAYS` in `H5Oprivate.h`, release 2.2.0.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct MessageFlags(u8);

impl MessageFlags {
    /// Wraps the flags byte stored by a message record.
    pub(crate) const fn new(byte: u8) -> Self {
        Self(byte)
    }

    /// Returns the flags byte stored by the message record.
    pub(crate) const fn get(self) -> u8 {
        self.0
    }

    /// Returns `true` if [`Self::SHARED`] is set.
    pub(crate) const fn is_shared(self) -> bool {
        self.contains(Self::SHARED)
    }

    /// Returns `true` if [`Self::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE`] is set.
    pub(crate) const fn fails_if_unknown_and_open_for_write(self) -> bool {
        self.contains(Self::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE)
    }

    /// Returns `true` if [`Self::FAIL_IF_UNKNOWN_ALWAYS`] is set.
    pub(crate) const fn fails_if_unknown_always(self) -> bool {
        self.contains(Self::FAIL_IF_UNKNOWN_ALWAYS)
    }

    /// Returns `true` if an unknown message must be rejected under `access_mode`.
    ///
    /// [`Self::FAIL_IF_UNKNOWN_ALWAYS`] applies in either access mode.
    /// [`Self::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE`] applies under [`AccessMode::ReadWrite`].
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

    /// Contains no flags.
    pub(crate) const NONE: Self = Self(0x00);

    /// Marks message data as constant.
    pub(crate) const CONSTANT: Self = Self(0x01);

    /// Marks a message as stored outside the object header.
    ///
    /// The message record body contains a reference to the shared message.
    pub(crate) const SHARED: Self = Self(0x02);

    /// Prevents the message from being moved into shared storage.
    pub(crate) const FORBID_SHARING: Self = Self(0x04);

    /// Requires an unknown message to be rejected while the file is open for writing.
    pub(crate) const FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE: Self = Self(0x08);

    /// Requires an unknown message to be marked when the object is modified.
    pub(crate) const MARK_IF_UNKNOWN: Self = Self(0x10);

    /// Marks a message whose object was modified by a library that did not recognize the message.
    pub(crate) const WAS_UNKNOWN: Self = Self(0x20);

    /// Allows the message to be moved into shared storage.
    pub(crate) const SHAREABLE: Self = Self(0x40);

    /// Requires an unknown message to be rejected in either access mode.
    pub(crate) const FAIL_IF_UNKNOWN_ALWAYS: Self = Self(0x80);

    const NAMED: [(Self, &'static str); 8] = [
        (Self::CONSTANT, "CONSTANT"),
        (Self::SHARED, "SHARED"),
        (Self::FORBID_SHARING, "FORBID_SHARING"),
        (
            Self::FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE,
            "FAIL_IF_UNKNOWN_AND_OPEN_FOR_WRITE",
        ),
        (Self::MARK_IF_UNKNOWN, "MARK_IF_UNKNOWN"),
        (Self::WAS_UNKNOWN, "WAS_UNKNOWN"),
        (Self::SHAREABLE, "SHAREABLE"),
        (Self::FAIL_IF_UNKNOWN_ALWAYS, "FAIL_IF_UNKNOWN_ALWAYS"),
    ];
}

impl BitOr for MessageFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Lists the flags that are set, as `MessageFlags(SHARED | FORBID_SHARING)`.
impl fmt::Debug for MessageFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MessageFlags(")?;

        let mut separator = "";
        for &(flag, name) in &Self::NAMED {
            if self.contains(flag) {
                f.write_str(separator)?;
                f.write_str(name)?;
                separator = " | ";
            }
        }

        f.write_str(")")
    }
}

const _: () = {
    assert!(core::mem::size_of::<MessageFlags>() == core::mem::size_of::<u8>());
    assert!(core::mem::align_of::<MessageFlags>() == core::mem::align_of::<u8>());
};

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
        for &(flag, _) in &MessageFlags::NAMED {
            assert_eq!(flags_set_in(flag), vec![flag]);
        }
    }

    #[test]
    fn every_bit_of_a_byte_stands_for_a_flag() {
        let all = MessageFlags::NAMED
            .iter()
            .map(|&(flag, _)| flag)
            .fold(MessageFlags::NONE, |all, flag| all | flag);

        assert_eq!(all.get(), u8::MAX);
        assert_eq!(
            flags_set_in(all),
            MessageFlags::NAMED
                .iter()
                .map(|&(flag, _)| flag)
                .collect::<Vec<_>>()
        );
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
        MessageFlags::NAMED
            .iter()
            .filter_map(|&(flag, _)| flags.contains(flag).then_some(flag))
            .collect()
    }
}
