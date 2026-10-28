//! File-format state for an editing session.

use alloc::vec::Vec;

use hdf5_pure_core::__private::PhysicalAddress;

use crate::address::AddressContext;
use crate::error::Error;
use crate::file_writer::{LENGTH_WIDTH, OFFSET_WIDTH};
use crate::superblock::{self, LocatedSuperblock, Superblock};
use crate::width::FormatWidths;

use super::allocation::AllocationPublication;

/// The superblock, address domain, and allocation publication state of an editable file.
///
/// The editor supports 8-byte offsets and lengths and a superblock located at its base address.
/// Widths, the address mapping, and the signature position are derived from the superblock.
pub(super) struct FileContext {
    superblock: Superblock,
    pub(super) allocations: AllocationPublication,
}

impl FileContext {
    /// Requires the editor's supported widths and canonical superblock position.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if either width differs from eight bytes, or if the
    /// superblock's physical position differs from its encoded base address.
    pub(super) fn new(located: LocatedSuperblock) -> Result<Self, Error> {
        let LocatedSuperblock {
            address,
            superblock,
            widths: _,
        } = located;
        if superblock.offset_size != OFFSET_WIDTH || superblock.length_size != LENGTH_WIDTH {
            return Err(Error::EditUnsupported(
                "only 8-byte offsets and lengths are supported for in-place editing",
            ));
        }
        if superblock.base_address.get() != address.get() {
            return Err(Error::EditUnsupported(
                "a file whose superblock is not located at its base address is not editable in place",
            ));
        }
        Ok(Self {
            superblock,
            allocations: AllocationPublication::new(),
        })
    }

    /// Replaces the session's superblock with a checked proposal.
    ///
    /// The caller completes the applicable publication or recovery steps before adoption.
    /// Rechecks the proposal's format and address domain against this context.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the proposal's base address, widths, or superblock
    /// version differ from the opened file's values.
    pub(super) fn adopt(&mut self, proposed: ProposedSuperblock) -> Result<(), Error> {
        self.require_format(&proposed.superblock)?;
        self.superblock = proposed.superblock;
        Ok(())
    }

    /// Checks a replacement superblock against the opened format and address domain.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the proposed base address, widths, or superblock
    /// version differ from the opened file's values.
    pub(super) fn propose(&self, superblock: Superblock) -> Result<ProposedSuperblock, Error> {
        self.require_format(&superblock)?;
        Ok(ProposedSuperblock { superblock })
    }

    /// Parses the published superblock for PAGE recovery.
    ///
    /// The caller compares `published` with the accepted file image before recovery. Only the
    /// root and end-of-file addresses may change.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if other encoded fields differ from the session's
    /// superblock. Returns [`Error::Format`] if parsing or serialization fails.
    pub(super) fn recover(&self, published: &[u8]) -> Result<ProposedSuperblock, Error> {
        let recovered = superblock::read(published)?.superblock;
        let mut expected = self.superblock.clone();
        expected.root_group_address = recovered.root_group_address;
        expected.eof_address = recovered.eof_address;
        if hdf5_pure_format::__private::serialize_superblock(&expected)? != published {
            return Err(Error::EditUnsupported(
                "PAGE superblock framing changed after a failed publication; \
                 close and reopen the file before editing again",
            ));
        }
        self.propose(recovered)
    }

    /// Returns the session's superblock with encoded address values.
    pub(super) const fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Returns the offset and length widths stored in the superblock.
    pub(super) const fn widths(&self) -> FormatWidths {
        FormatWidths {
            offsets: self.superblock.offset_size,
            lengths: self.superblock.length_size,
        }
    }

    /// Returns the mapping between stored addresses and physical file positions.
    pub(super) const fn addresses(&self) -> AddressContext {
        AddressContext::from_effective_base(self.superblock.base_address)
    }

    /// Returns the physical position of the superblock signature.
    pub(super) const fn signature(&self) -> PhysicalAddress {
        PhysicalAddress::new(self.superblock.base_address.get())
    }

    fn require_format(&self, superblock: &Superblock) -> Result<(), Error> {
        if superblock.base_address != self.superblock.base_address
            || superblock.offset_size != self.superblock.offset_size
            || superblock.length_size != self.superblock.length_size
            || superblock.version() != self.superblock.version()
        {
            return Err(Error::EditUnsupported(
                "a proposed superblock changes the opened file's format or address domain",
            ));
        }
        Ok(())
    }
}

/// A superblock with its base address, widths, and version checked against a file context.
///
/// [`FileContext::adopt`] rechecks these fields against the destination context.
#[derive(Debug)]
pub(super) struct ProposedSuperblock {
    superblock: Superblock,
}

impl ProposedSuperblock {
    /// Serializes the checked superblock for a publication write.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Format`] if the superblock cannot be encoded.
    pub(super) fn encode(self) -> Result<SuperblockPublication, Error> {
        let bytes = hdf5_pure_format::__private::serialize_superblock(&self.superblock)?;
        Ok(SuperblockPublication {
            proposed: self,
            bytes,
        })
    }
}

/// A checked superblock paired with its serialized bytes.
pub(super) struct SuperblockPublication {
    proposed: ProposedSuperblock,
    bytes: Vec<u8>,
}

impl SuperblockPublication {
    /// Discards the serialized bytes and returns the checked superblock.
    pub(super) fn into_proposed(self) -> ProposedSuperblock {
        self.proposed
    }

    /// Returns the complete serialized superblock for a publication write.
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::superblock::v2;
    use test_util::widths::Widths;

    use crate::address::{BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt};
    use crate::error::FormatError;
    use crate::superblock;
    use crate::width::{LengthWidth, OffsetWidth};
    use crate::{SuperblockStatusFlags, SuperblockVersion};

    use super::{Error, FileContext, LocatedSuperblock, PhysicalAddress};

    #[rstest]
    #[case::plain(0)]
    #[case::userblock(512)]
    fn context_derives_the_opened_address_domain(#[case] base: u64) {
        let context = FileContext::new(located(base, Widths::new(8, 8))).unwrap();
        assert_eq!(context.signature().get(), base);
        assert_eq!(context.widths().offsets, OffsetWidth::Eight);
        assert_eq!(context.widths().lengths, LengthWidth::Eight);
        assert_eq!(context.superblock().base_address.get(), base);
        let physical = context.addresses().resolve(StoredAddress::new(96)).unwrap();
        assert_eq!(physical.get(), base + 96);
        assert_eq!(
            context.addresses().relative(physical.get()).unwrap().get(),
            96
        );
    }

    #[rstest]
    #[case::offset_width(Widths::new(4, 8))]
    #[case::length_width(Widths::new(8, 4))]
    fn context_rejects_unsupported_edit_widths(#[case] widths: Widths) {
        let Err(Error::EditUnsupported(reason)) = FileContext::new(located(0, widths)) else {
            panic!("expected unsupported edit widths");
        };
        assert_eq!(
            reason,
            "only 8-byte offsets and lengths are supported for in-place editing"
        );
    }

    #[test]
    fn context_rejects_a_relocated_superblock() {
        let mut located = located(512, Widths::new(8, 8));
        located.address = PhysicalAddress::new(1024);
        let Err(Error::EditUnsupported(reason)) = FileContext::new(located) else {
            panic!("expected unsupported relocated superblock");
        };
        assert_eq!(
            reason,
            "a file whose superblock is not located at its base address is not editable in place"
        );
    }

    #[rstest]
    #[case::base(Some(BaseAddress::ZERO), None, None, None)]
    #[case::offset_width(None, Some(OffsetWidth::Four), None, None)]
    #[case::length_width(None, None, Some(LengthWidth::Four), None)]
    #[case::version(None, None, None, Some(SuperblockVersion::Three))]
    fn proposals_preserve_the_opened_format(
        #[case] base: Option<BaseAddress>,
        #[case] offsets: Option<OffsetWidth>,
        #[case] lengths: Option<LengthWidth>,
        #[case] version: Option<SuperblockVersion>,
    ) {
        let context = FileContext::new(located(512, Widths::new(8, 8))).unwrap();
        let before = context.superblock().clone();
        let mut proposed = before.clone();
        proposed.base_address = base.unwrap_or(proposed.base_address);
        proposed.offset_size = offsets.unwrap_or(proposed.offset_size);
        proposed.length_size = lengths.unwrap_or(proposed.length_size);
        proposed
            .set_version(version.unwrap_or(proposed.version()))
            .unwrap();
        let Error::EditUnsupported(reason) = context.propose(proposed).unwrap_err() else {
            panic!("expected incompatible publication format");
        };
        assert_eq!(
            reason,
            "a proposed superblock changes the opened file's format or address domain"
        );
        assert_eq!(context.superblock(), &before);
        assert_eq!(context.signature().get(), 512);
    }

    #[test]
    fn adoption_preserves_address_conversion_errors() {
        let mut context = FileContext::new(located(512, Widths::new(8, 8))).unwrap();
        let mut proposed = context.superblock().clone();
        proposed.root_group_address = StoredAddress::new(128);
        proposed.eof_address = 4096;
        let proposed = context.propose(proposed).unwrap();
        context.adopt(proposed).unwrap();
        assert_eq!(
            context
                .addresses()
                .absolute(context.superblock().root_group_address)
                .unwrap(),
            640
        );
        assert_eq!(context.superblock().eof_address, 4096);
        assert_eq!(
            context.addresses().relative(511).unwrap_err(),
            FormatError::AddressBelowBase {
                address: 511,
                base: 512
            }
        );
        assert_eq!(
            context
                .addresses()
                .absolute(StoredAddress::new(u64::MAX))
                .unwrap_err(),
            FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 512
            }
        );
    }

    #[test]
    fn encoding_preserves_context_until_adoption() {
        let mut context = FileContext::new(located(512, Widths::EIGHT)).unwrap();
        let before = context.superblock().clone();
        let mut proposed = before.clone();
        proposed.root_group_address = StoredAddress::new(192);
        proposed.eof_address = 8192;
        proposed
            .set_consistency_flags(SuperblockStatusFlags::write_marked())
            .unwrap();
        let publication = context.propose(proposed.clone()).unwrap().encode().unwrap();
        assert_eq!(context.superblock(), &before);
        assert_eq!(
            publication.bytes(),
            v2::Superblock::new(Widths::EIGHT)
                .base_address(512)
                .root_header_address(192)
                .eof_address(8192)
                .consistency_flags(
                    SuperblockStatusFlags::write_marked()
                        .bits()
                        .try_into()
                        .unwrap()
                )
                .build()
        );
        context.adopt(publication.into_proposed()).unwrap();
        assert_eq!(context.superblock(), &proposed);
        assert_eq!(
            context
                .addresses()
                .absolute(proposed.root_group_address)
                .unwrap(),
            704
        );
    }

    #[test]
    fn adoption_rejects_a_proposal_from_another_address_domain() {
        let source = FileContext::new(located(512, Widths::EIGHT)).unwrap();
        let proposed = source.propose(source.superblock().clone()).unwrap();
        let mut target = FileContext::new(located(0, Widths::EIGHT)).unwrap();
        let before = target.superblock().clone();
        let Error::EditUnsupported(reason) = target.adopt(proposed).unwrap_err() else {
            panic!("expected incompatible publication format");
        };
        assert_eq!(
            reason,
            "a proposed superblock changes the opened file's format or address domain"
        );
        assert_eq!(target.superblock(), &before);
    }

    #[rstest]
    #[case::root_and_eof(None, None, None, None, false, false)]
    #[case::base(Some(BaseAddress::ZERO), None, None, None, false, false)]
    #[case::offset_width(None, Some(OffsetWidth::Four), None, None, false, false)]
    #[case::length_width(None, None, Some(LengthWidth::Four), None, false, false)]
    #[case::version(None, None, None, Some(SuperblockVersion::Three), false, false)]
    #[case::flags(None, None, None, None, true, false)]
    #[case::extension(None, None, None, None, false, true)]
    fn recovery_adopts_only_root_and_eof_changes(
        #[case] base: Option<BaseAddress>,
        #[case] offsets: Option<OffsetWidth>,
        #[case] lengths: Option<LengthWidth>,
        #[case] version: Option<SuperblockVersion>,
        #[case] flags: bool,
        #[case] extension: bool,
    ) {
        let mut context = FileContext::new(located(512, Widths::EIGHT)).unwrap();
        let before = context.superblock().clone();
        let mut recovered = before.clone();
        recovered.root_group_address = StoredAddress::new(192);
        recovered.eof_address = 8192;
        recovered.base_address = base.unwrap_or(recovered.base_address);
        recovered.offset_size = offsets.unwrap_or(recovered.offset_size);
        recovered.length_size = lengths.unwrap_or(recovered.length_size);
        recovered
            .set_version(version.unwrap_or(recovered.version()))
            .unwrap();
        if flags {
            recovered
                .set_consistency_flags(SuperblockStatusFlags::write_marked())
                .unwrap();
        }
        if extension {
            recovered.superblock_extension_address = Some(StoredAddress::new(256));
        }
        let published = hdf5_pure_format::__private::serialize_superblock(&recovered).unwrap();
        let result = context.recover(&published);
        assert_eq!(context.superblock(), &before);
        if base.is_some()
            || offsets.is_some()
            || lengths.is_some()
            || version.is_some()
            || flags
            || extension
        {
            let Error::EditUnsupported(reason) = result.unwrap_err() else {
                panic!("expected changed publication framing");
            };
            assert_eq!(
                reason,
                "PAGE superblock framing changed after a failed publication; close and reopen the file before editing again"
            );
            assert_eq!(context.superblock(), &before);
        } else {
            context.adopt(result.unwrap()).unwrap();
            recovered.checksum = Some(u32::from_le_bytes(*published.last_chunk().unwrap()));
            assert_eq!(context.superblock(), &recovered);
            assert_eq!(context.signature().get(), 512);
        }
    }

    #[rstest]
    #[case::missing_signature(Some((0, 0)), 48, FormatError::SignatureNotFound)]
    #[case::version(Some((8, 4)), 48, FormatError::UnsupportedVersion(4))]
    #[case::offset_width(Some((9, 3)), 48, FormatError::InvalidOffsetSize(3))]
    #[case::length_width(Some((10, 5)), 48, FormatError::InvalidLengthSize(5))]
    #[case::candidate_truncated(
        None,
        10,
        FormatError::UnexpectedSourceEof { expected: 15, available: 10 }
    )]
    #[case::superblock_truncated(
        None,
        47,
        FormatError::UnexpectedSourceEof { expected: 48, available: 47 }
    )]
    fn recovery_preserves_parsing_errors(
        #[case] changed: Option<(usize, u8)>,
        #[case] len: usize,
        #[case] expected: FormatError,
    ) {
        let context = FileContext::new(located(512, Widths::EIGHT)).unwrap();
        let before = context.superblock().clone();
        let mut published = v2::Superblock::new(Widths::EIGHT)
            .base_address(512)
            .root_header_address(192)
            .eof_address(8192)
            .build();
        if let Some((at, value)) = changed {
            published[at] = value;
        }
        published.truncate(len);
        let Error::Format(error) = context.recover(&published).unwrap_err() else {
            panic!("expected superblock parsing error");
        };
        assert_eq!(error, expected);
        assert_eq!(context.superblock(), &before);
    }

    #[test]
    fn recovery_rejects_a_checksum_mismatch() {
        let context = FileContext::new(located(512, Widths::EIGHT)).unwrap();
        let before = context.superblock().clone();
        let mut published = v2::Superblock::new(Widths::EIGHT)
            .base_address(512)
            .root_header_address(192)
            .eof_address(8192)
            .build();
        #[cfg(feature = "checksum")]
        let computed = u32::from_le_bytes(*published.last_chunk().unwrap());
        *published.last_mut().unwrap() ^= 0xFF;
        let error = context.recover(&published).unwrap_err();
        #[cfg(feature = "checksum")]
        {
            let Error::Format(error) = error else {
                panic!("expected superblock checksum mismatch");
            };
            assert_eq!(
                error,
                FormatError::ChecksumMismatch {
                    expected: computed ^ 0xFF00_0000,
                    computed,
                }
            );
        }
        #[cfg(not(feature = "checksum"))]
        {
            let Error::EditUnsupported(reason) = error else {
                panic!("expected changed publication framing");
            };
            assert_eq!(
                reason,
                "PAGE superblock framing changed after a failed publication; close and reopen the file before editing again"
            );
        }
        assert_eq!(context.superblock(), &before);
    }

    fn located(base: u64, widths: Widths) -> LocatedSuperblock {
        let bytes = v2::Superblock::new(widths)
            .base_address(base)
            .eof_address(4096)
            .root_header_address(96)
            .build();
        let mut located = superblock::read(bytes.as_slice()).unwrap();
        located.address = PhysicalAddress::new(base);
        located
    }
}
