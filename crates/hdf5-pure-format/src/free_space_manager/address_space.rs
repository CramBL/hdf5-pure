use crate::width::OffsetWidth;

/// The "Size of Address Space" field of a free-space manager header: the number of bits of an
/// address in the space the manager tracks, 1 to 64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AddressSpaceBits(u8);

impl AddressSpaceBits {
    /// Returns the address space of `bits` bits, or `None` if `bits` is 0 or more than 64.
    pub(super) fn new(bits: u16) -> Option<Self> {
        u8::try_from(bits)
            .ok()
            .filter(|bits| (1..=64).contains(bits))
            .map(Self)
    }

    /// Returns the "Size of Address Space" the serializer writes in a file with addresses of
    /// `offsets`, one bit less than the width of an address.
    pub(super) fn written(offsets: OffsetWidth) -> Self {
        Self(offsets.get() * 8 - 1)
    }

    pub(super) fn get(self) -> u16 {
        u16::from(self.0)
    }

    /// Returns the width in bytes of a section offset in this address space.
    pub(super) fn offset_width(self) -> usize {
        usize::from(self.0).div_ceil(8)
    }
}
