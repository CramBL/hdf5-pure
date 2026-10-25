//! File-format state for an editing session.

use hdf5_pure_core::__private::PhysicalAddress;

use crate::SuperblockStatusFlags;
use crate::address::AddressContext;
use crate::error::Error;
use crate::file_writer::{LENGTH_WIDTH, OFFSET_WIDTH};
use crate::superblock::{LocatedSuperblock, Superblock};
use crate::width::FormatWidths;

/// The superblock and address domain of an editable file.
///
/// The editor supports 8-byte offsets and lengths and a superblock located at its base address.
/// Widths, the address mapping, and the signature position are derived from the superblock.
pub(super) struct FileContext {
    superblock: Superblock,
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
        Ok(Self { superblock })
    }

    /// Replaces the session's superblock while preserving its format and address domain.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if the proposed base address, widths, or superblock
    /// version differ from the opened file's values.
    pub(super) fn adopt(&mut self, superblock: Superblock) -> Result<(), Error> {
        if superblock.base_address != self.superblock.base_address
            || superblock.offset_size != self.superblock.offset_size
            || superblock.length_size != self.superblock.length_size
            || superblock.version != self.superblock.version
        {
            return Err(Error::EditUnsupported(
                "a proposed superblock changes the opened file's format or address domain",
            ));
        }
        self.superblock = superblock;
        Ok(())
    }

    /// Updates the in-memory consistency flags.
    pub(super) fn set_consistency_flags(&mut self, flags: SuperblockStatusFlags) {
        self.superblock.consistency_flags = flags;
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
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use test_util::superblock::v2;
    use test_util::widths::Widths;

    use crate::SuperblockVersion;
    use crate::address::{BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt};
    use crate::error::FormatError;
    use crate::superblock;
    use crate::width::{LengthWidth, OffsetWidth};

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
    fn adoption_preserves_the_opened_format(
        #[case] base: Option<BaseAddress>,
        #[case] offsets: Option<OffsetWidth>,
        #[case] lengths: Option<LengthWidth>,
        #[case] version: Option<SuperblockVersion>,
    ) {
        let mut context = FileContext::new(located(512, Widths::new(8, 8))).unwrap();
        let before = context.superblock().clone();
        let mut proposed = before.clone();
        proposed.base_address = base.unwrap_or(proposed.base_address);
        proposed.offset_size = offsets.unwrap_or(proposed.offset_size);
        proposed.length_size = lengths.unwrap_or(proposed.length_size);
        proposed.version = version.unwrap_or(proposed.version);
        let Error::EditUnsupported(reason) = context.adopt(proposed).unwrap_err() else {
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
