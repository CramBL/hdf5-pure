use crate::access_mode::AccessMode;
use crate::address::BaseAddress;
use crate::width::FormatWidths;

/// The access mode, field widths, and base address shared by all chunks in a parse.
#[derive(Clone, Copy)]
pub(super) struct ParseContext {
    pub(super) access_mode: AccessMode,
    pub(super) widths: FormatWidths,
    pub(super) base_address: BaseAddress,
}
