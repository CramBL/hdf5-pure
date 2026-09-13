//! The access mode an object header parse is made under.

/// Read-only or read-write access to a file's objects.
///
/// The specification requires a decoder that cannot name a message's type to reject the object
/// while the file is open for write access, bit 3 of the message record's "Header Message #n
/// Flags" field, defined in "Version 1 Data Object Header Prefix" of the [format specification,
/// version 4.0][spec]. A caller passes the mode its file is open under to every object header
/// parse of that file, or [`AccessMode::ReadWrite`] to have a read-only open reject a message
/// only a decoder with write access must understand.
/// [`must_be_understood`](crate::message_flags::MessageFlags::must_be_understood) tests that bit
/// under [`AccessMode::ReadWrite`] alone.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_prefix_one
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AccessMode {
    /// The file is open for reading alone.
    ReadOnly,
    /// The file is open for editing, or a read of it rejects a message only a
    /// decoder with write access must understand.
    ReadWrite,
}

impl AccessMode {
    /// Returns `true` for [`AccessMode::ReadWrite`].
    pub(crate) const fn is_read_write(self) -> bool {
        matches!(self, Self::ReadWrite)
    }
}
