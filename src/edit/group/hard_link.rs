//! Checked hard Link records with fixed-width stored target fields.

use core::ops::Range;

use hdf5_pure_format::__private::MessageRecordLayout;

use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::error::Error;
use crate::error::FormatError;
use crate::error::OBJECT_HEADER_MESSAGE_MAX;
use crate::file_writer;
use crate::link_message::LinkMessage;
use crate::message_type::MessageType;
use crate::object_path::LinkNameBuf;
use crate::width::OffsetWidth;

/// An encoded hard Link record whose body fits the object-header message limit.
///
/// The target address occupies a checked byte range at the selected offset width. Patching it
/// preserves the record length, name, and link creation index.
#[derive(Debug)]
pub(in crate::edit) struct PreparedHardLinkRecord {
    offsets: OffsetWidth,
    record: Vec<u8>,
    target: Range<usize>,
    layout: MessageRecordLayout,
}

impl PreparedHardLinkRecord {
    /// Encodes a hard link with a zero target after checking its message-body size.
    ///
    /// `creation_order` is the link's creation index. The caller supplies the object-header
    /// record layout separately. The target field occupies `offsets` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the encoded body exceeds
    /// [`OBJECT_HEADER_MESSAGE_MAX`]. Returns [`Error::Format`] if the encoded target-address
    /// range cannot be read.
    pub(in crate::edit) fn new(
        offsets: OffsetWidth,
        layout: MessageRecordLayout,
        name: LinkNameBuf,
        creation_order: Option<u64>,
    ) -> Result<Self, Error> {
        let mut link = file_writer::make_link(name.as_str(), StoredAddress::new(0));
        link.creation_order = creation_order;
        let body = link.serialize(offsets)?;
        if body.len() > OBJECT_HEADER_MESSAGE_MAX {
            return Err(Error::EditUnsupported(
                "a new hard link message exceeds the object-header message size limit",
            ));
        }
        let target = LinkMessage::hard_link_address_range(&body, offsets)?.ok_or_else(|| {
            FormatError::Internal("a prepared hard link has no target address".into())
        })?;
        let target = layout.prefix_len() + target.start..layout.prefix_len() + target.end;
        Ok(Self {
            offsets,
            record: layout.record(MessageType::LINK, &body),
            target,
            layout,
        })
    }

    /// Writes a stored target address into the prepared record.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::AddressTooLarge`] if `address` exceeds the selected offset width,
    /// [`FormatError::DefinedAddressIsUndefined`] if it equals the undefined marker, or
    /// [`FormatError::Internal`] if the target range is outside the record. Preserves the record
    /// on a width error.
    pub(in crate::edit) fn patch_target(&mut self, address: StoredAddress) -> Result<(), Error> {
        let target = self.record.get_mut(self.target.clone()).ok_or_else(|| {
            FormatError::Internal("a prepared hard link target is outside its record".into())
        })?;
        hdf5_pure_format::__private::write_defined_stored_address_at(
            target,
            0,
            address,
            self.offsets,
        )?;
        Ok(())
    }

    /// Borrows the encoded name from the prepared record's message body.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the record is shorter than its prefix. Propagates
    /// errors from reading the encoded Link name.
    pub(in crate::edit) fn name_bytes(&self) -> Result<&[u8], Error> {
        let body = self.record.get(self.layout.prefix_len()..).ok_or_else(|| {
            FormatError::Internal("a prepared hard link has no message body".into())
        })?;
        Ok(LinkMessage::name_bytes(body)?)
    }

    /// Returns the encoded record, including its object-header message prefix.
    pub(in crate::edit) fn as_bytes(&self) -> &[u8] {
        &self.record
    }
}

#[cfg(test)]
mod tests {
    use hdf5_pure_format::__private::MessageRecordLayout;
    use rstest::rstest;
    use test_util::link_message::HardLink;
    use test_util::object_header::Message;
    use test_util::object_header::MessageType as FixtureMessage;
    use test_util::object_header::v2;
    use test_util::object_header::v2::HeaderFlags;
    use test_util::widths::Widths;

    use super::PreparedHardLinkRecord;
    use crate::FormatError;
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::error::Error;
    use crate::message_type::MessageType;
    use crate::object_path::LinkNameBuf;
    use crate::width::OffsetWidth;

    #[rstest]
    #[case::two(OffsetWidth::Two, 65_534)]
    #[case::four(OffsetWidth::Four, 4_294_967_294)]
    #[case::eight(OffsetWidth::Eight, u64::MAX - 1)]
    fn prepared_hard_links_use_the_selected_target_width(
        #[case] offsets: OffsetWidth,
        #[case] address: u64,
        #[values(HeaderFlags::default(), HeaderFlags::TRACKS_CREATION_ORDER)] flags: HeaderFlags,
    ) {
        let mut record = PreparedHardLinkRecord::new(
            offsets,
            MessageRecordLayout::from_header_flags(flags.0),
            LinkNameBuf::new("target").unwrap(),
            Some(17),
        )
        .unwrap();
        record.patch_target(StoredAddress::new(address)).unwrap();
        let expected = HardLink::new("target", address)
            .creation_order(17)
            .build(Widths::new(usize::from(offsets.get()), 8));
        assert_eq!(
            record.as_bytes(),
            v2::message_record(&Message::new(FixtureMessage::LINK, &expected), flags)
        );
    }

    #[rstest]
    #[case::two_undefined(OffsetWidth::Two, 65_535, FormatError::DefinedAddressIsUndefined { address: 65_535, offset_size: 2 })]
    #[case::two_overflow(OffsetWidth::Two, 65_536, FormatError::AddressTooLarge { address: 65_536, offset_size: 2 })]
    #[case::four_undefined(OffsetWidth::Four, 4_294_967_295, FormatError::DefinedAddressIsUndefined { address: 4_294_967_295, offset_size: 4 })]
    #[case::four_overflow(OffsetWidth::Four, 4_294_967_296, FormatError::AddressTooLarge { address: 4_294_967_296, offset_size: 4 })]
    #[case::eight_undefined(OffsetWidth::Eight, u64::MAX, FormatError::DefinedAddressIsUndefined { address: u64::MAX, offset_size: 8 })]
    fn prepared_hard_links_preserve_records_on_unrepresentable_targets(
        #[case] offsets: OffsetWidth,
        #[case] address: u64,
        #[case] expected: FormatError,
    ) {
        let mut record = PreparedHardLinkRecord::new(
            offsets,
            MessageRecordLayout::PLAIN,
            LinkNameBuf::new("target").unwrap(),
            None,
        )
        .unwrap();
        let before = record.as_bytes().to_vec();
        let error = record
            .patch_target(StoredAddress::new(address))
            .unwrap_err();
        let Error::Format(actual) = error else {
            panic!("expected Format, got {error:?}");
        };
        assert_eq!(actual, expected);
        assert_eq!(record.as_bytes(), before);
    }

    #[rstest]
    #[case::untracked(65_523, None)]
    #[case::tracked(65_515, Some(17))]
    fn prepared_hard_links_accept_the_maximum_body_length(
        #[case] byte_len: usize,
        #[case] creation_order: Option<u64>,
        #[values(NameEncoding::Ascii, NameEncoding::Utf8)] encoding: NameEncoding,
        #[values(HeaderFlags::default(), HeaderFlags::TRACKS_CREATION_ORDER)] flags: HeaderFlags,
    ) {
        let name = encoding.name(byte_len);
        assert_eq!(name.len(), byte_len);
        let layout = MessageRecordLayout::from_header_flags(flags.0);
        let record = PreparedHardLinkRecord::new(
            OffsetWidth::Eight,
            layout,
            LinkNameBuf::new(&name).unwrap(),
            creation_order,
        )
        .unwrap();
        let mut body = HardLink::new(&name, 0).name_size_width(2);
        if let Some(index) = creation_order {
            body = body.creation_order(index);
        }
        let body = body.build(Widths::EIGHT);
        assert_eq!(body.len(), MAX_MESSAGE_BODY_LENGTH);
        assert_eq!(
            record.as_bytes(),
            v2::message_record(&Message::new(FixtureMessage::LINK, &body), flags)
        );
        let parsed = layout.next_message(record.as_bytes(), 0).unwrap().unwrap();
        assert_eq!(parsed.msg_type, MessageType::LINK);
        assert_eq!(parsed.body, body);
        assert_eq!(
            parsed.creation_index,
            flags.tracks_creation_order().then_some(0)
        );
    }

    #[rstest]
    #[case::untracked(65_524, None)]
    #[case::tracked(65_516, Some(17))]
    #[case::historical(65_532, None)]
    fn prepared_hard_links_reject_oversized_bodies(
        #[case] byte_len: usize,
        #[case] creation_order: Option<u64>,
        #[values(NameEncoding::Ascii, NameEncoding::Utf8)] encoding: NameEncoding,
        #[values(HeaderFlags::default(), HeaderFlags::TRACKS_CREATION_ORDER)] flags: HeaderFlags,
    ) {
        let name = encoding.name(byte_len);
        assert_eq!(name.len(), byte_len);
        let error = PreparedHardLinkRecord::new(
            OffsetWidth::Eight,
            MessageRecordLayout::from_header_flags(flags.0),
            LinkNameBuf::new(&name).unwrap(),
            creation_order,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                Error::EditUnsupported(
                    "a new hard link message exceeds the object-header message size limit"
                )
            ),
            "{error:?}"
        );
    }

    #[rstest]
    #[case::single_byte(1, 1)]
    #[case::one_byte_maximum(255, 1)]
    #[case::two_byte_minimum(256, 2)]
    #[case::maximum(65_515, 2)]
    fn prepared_hard_link_patching_preserves_every_other_byte(
        #[case] byte_len: usize,
        #[case] name_size_width: usize,
        #[values(None, Some(0x0102_0304_0506_0708))] creation_order: Option<u64>,
        #[values(NameEncoding::Ascii, NameEncoding::Utf8)] encoding: NameEncoding,
        #[values(HeaderFlags::default(), HeaderFlags::TRACKS_CREATION_ORDER)] flags: HeaderFlags,
        #[values(0x0102_0304_0506_0708, u64::MAX - 1)] address: u64,
    ) {
        let name = encoding.name(byte_len);
        assert_eq!(name.len(), byte_len);
        let mut record = PreparedHardLinkRecord::new(
            OffsetWidth::Eight,
            MessageRecordLayout::from_header_flags(flags.0),
            LinkNameBuf::new(&name).unwrap(),
            creation_order,
        )
        .unwrap();
        let before = record.as_bytes().to_vec();
        record.patch_target(StoredAddress::new(address)).unwrap();
        let target = record.target.clone();
        assert_eq!(record.as_bytes().len(), before.len());
        assert_eq!(&record.as_bytes()[..target.start], &before[..target.start]);
        assert_eq!(&record.as_bytes()[target.end..], &before[target.end..]);
        assert_eq!(&record.as_bytes()[target], &address.to_le_bytes());
        let mut expected = HardLink::new(&name, address).name_size_width(name_size_width);
        if let Some(index) = creation_order {
            expected = expected.creation_order(index);
        }
        assert_eq!(
            record.as_bytes(),
            v2::message_record(
                &Message::new(FixtureMessage::LINK, &expected.build(Widths::EIGHT)),
                flags,
            )
        );
    }

    #[derive(Clone, Copy, Debug)]
    enum NameEncoding {
        Ascii,
        Utf8,
    }

    impl NameEncoding {
        fn name(self, byte_len: usize) -> String {
            match self {
                Self::Ascii => "x".repeat(byte_len),
                Self::Utf8 => {
                    let mut name = "é".repeat(byte_len / "é".len());
                    if byte_len % "é".len() != 0 {
                        name.push('x');
                    }
                    name
                }
            }
        }
    }

    // "Version 2 Data Object Header Prefix" in the format specification, version 4.0,
    // defines a two-byte message-body length field.
    const MAX_MESSAGE_BODY_LENGTH: usize = 65_535;
}
