//! The free-space manager's structures: section
//! `subsec_fmt4_infra_freespaceindex`, version 4.0.

/// The free-space manager header, which states how much space a manager
/// tracks and where its sections live.
pub const SIGNATURE: &[u8; 4] = b"FSHD";

/// The section-info block the header points at, which holds the free sections
/// themselves.
pub const SECTIONS_SIGNATURE: &[u8; 4] = b"FSSE";
