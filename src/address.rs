//! The superblock base address, as a type distinct from the addresses it shifts.
//!
//! # Why this exists
//!
//! The format specification says of the superblock's base address that "unless
//! otherwise noted, all other file addresses are relative to this base address".
//! A userblock makes that base non-zero — 512 bytes for every `.mat` file this
//! crate writes — so an HDF5 file holds addresses in *two* frames at once:
//!
//! * a **stored** address, as written in metadata, relative to the base;
//! * an **absolute** byte position in the file, which is what a
//!   [`Source`](crate::source::Source) reads at.
//!
//! Both are `u64`, both are usually equal (a plain file has a base of zero), and
//! the compiler has nothing to say about mixing them. The result is a defect
//! class rather than a defect: a reader that forgets the base lands inside the
//! userblock, and every test written against a base-0 file passes anyway.
//! `crates/crosscheck/tests/userblock_base_address.rs` was written after that happened
//! three separate times — dense attributes, object header continuations, and
//! dense link storage — and it covers the ground its four rows name, which is
//! not the same as covering the ground.
//!
//! [`BaseAddress`] does not make the two frames distinct types; addresses stay
//! `u64` throughout the crate. What it makes distinct is the **base itself**,
//! which is the operand every one of those defects got wrong. That buys three
//! things a bare `u64` did not give:
//!
//! * A base cannot be passed where an address is expected, or an address where a
//!   base is expected. Both directions were previously a silent argument swap in
//!   functions that take `(address, ..., base_address)` — the shape of nearly
//!   every parser in this crate.
//! * Both conversions are **checked, and named**: 46 call sites of
//!   [`BaseAddress::absolute`] and 18 of [`BaseAddress::relative`], plus their
//!   test fixtures. Most replaced a hand-written `checked_add(base)` chain, but
//!   **seven unchecked additions and eighteen unchecked subtractions** were in
//!   that set, each of which panics in a debug build and wraps in a release one.
//!   Two were the `stored_addr + base` in [`group_v2`](crate::group_v2)'s two
//!   path resolvers, now the single [`ChildLookup::of`](crate::group_v2::ChildLookup)
//!   that both share; `a_link_target_that_overflows_the_base_address_is_refused`
//!   covers it. The rest are in the write engine, where an absolute address goes
//!   back to stored form on the way into a link message or a superblock. Three
//!   additions of a base survive unconverted, all inside `debug_assert_eq!` in
//!   [`file_writer`](crate::file_writer), whose bodies a release build does not
//!   compile.
//! * "This code does not need the base" becomes [`BaseAddress::ZERO`]. Five call
//!   sites in the crate's code paths make that claim, so a `grep` finds all five
//!   — where a literal `0` argument was indistinguishable from the other zeroes
//!   on the line. (Another 34 are test fixtures satisfying a signature with a
//!   base they do not care about.)
//!
//! # What it deliberately does not do
//!
//! The read path resolves the two frames two different ways, and this type is
//! neutral between them. Object headers and group entries add the base to each
//! address as they parse it ([`absolute`](BaseAddress::absolute)); raw data,
//! chunk indices, and dense attribute storage instead *frame the file* at the
//! base — [`frame`](crate::source::frame) for a buffer,
//! [`BaseOffsetSource`](crate::source::BaseOffsetSource) for a stream — and read
//! stored addresses against that shifted view directly. Both are correct, and
//! which one a given parser is owed is a fact about the parser, not about the
//! base. A newtype for stored-versus-absolute *addresses* would encode that too;
//! it would also have to propagate through every chunk address, free-space
//! section, and index element in the crate, which is a far larger change than
//! this one and is not attempted here.

use crate::error::FormatError;

/// The byte offset at which a file's HDF5 image begins, as reported by
/// [`Superblock::base_address`](crate::Superblock::base_address).
///
/// Zero for a plain file, and the userblock size for a file that has one — 512
/// bytes for every `.mat` file this crate writes. Every address stored in a
/// file's metadata is relative to it, so an absolute position in the file is
/// the stored address plus this. [`get`](Self::get) is the number.
///
/// It is a type of its own rather than a `u64` because the base and the
/// addresses it shifts are both file offsets that mean different things, and
/// mixing them is a defect this crate hit three separate times: a reader that
/// forgets the base lands inside the userblock, and every test written against
/// a file without one passes anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaseAddress(u64);

impl BaseAddress {
    /// The base address of a file with no userblock, where a stored address and
    /// an absolute file offset are the same number.
    ///
    /// Written out at every call site that passes no base, so that "this parser
    /// is already reading a base-framed view" and "this file cannot have a
    /// userblock" are claims a reader can find and check, rather than a `0`
    /// among the other arguments.
    pub(crate) const ZERO: Self = Self(0);

    /// Take a base address read from a superblock.
    pub(crate) const fn new(base: u64) -> Self {
        Self(base)
    }

    /// The base as a plain integer: zero for a plain file, the userblock size
    /// for a file that has one.
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Whether this file's stored addresses are already absolute.
    ///
    /// The two framing helpers are the identity in this case, and several paths
    /// take a cheaper route through it (a streaming read need not wrap its
    /// source at all); a `base == 0` test spelled by hand reads as a magic
    /// number where this reads as the question being asked.
    pub(crate) const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// The absolute file position of `stored`, an address as written in metadata.
    ///
    /// [`FormatError::OffsetOverflow`] if the sum exceeds `u64`, which a
    /// malformed file can arrange: both operands are file-derived.
    pub(crate) fn absolute(self, stored: u64) -> Result<u64, FormatError> {
        stored
            .checked_add(self.0)
            .ok_or(FormatError::OffsetOverflow {
                offset: stored,
                length: self.0,
            })
    }

    /// The stored (base-relative) form of the absolute file position `at`, which
    /// is what metadata naming that position must hold.
    ///
    /// [`FormatError::AddressBelowBase`] if `at` is below the base, i.e. inside
    /// the userblock, where no HDF5 structure can live. Every call site this
    /// replaced subtracted without checking, so such an input wrapped to a
    /// near-`u64::MAX` address in a release build and panicked in a debug one.
    pub(crate) fn relative(self, at: u64) -> Result<u64, FormatError> {
        at.checked_sub(self.0).ok_or(FormatError::AddressBelowBase {
            address: at,
            base: self.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the type: a round trip through both conversions is the
    /// identity, for a base that shifts and a base that does not.
    #[test]
    fn the_two_conversions_invert_each_other() {
        for base in [BaseAddress::ZERO, BaseAddress::new(512)] {
            for stored in [0u64, 1, 4096, u32::MAX as u64] {
                let at = base.absolute(stored).unwrap();
                assert_eq!(base.relative(at).unwrap(), stored, "base {base:?}");
            }
        }
    }

    /// A base of zero leaves an address alone in both directions — which is what
    /// makes a forgotten base invisible on every plain file, and why the type
    /// exists.
    #[test]
    fn a_zero_base_is_the_identity() {
        assert!(BaseAddress::ZERO.is_zero());
        assert_eq!(BaseAddress::ZERO.absolute(1234).unwrap(), 1234);
        assert_eq!(BaseAddress::ZERO.relative(1234).unwrap(), 1234);
    }

    /// A userblock shifts by exactly its size, in both directions.
    #[test]
    fn a_userblock_shifts_by_its_size() {
        let base = BaseAddress::new(512);
        assert!(!base.is_zero());
        assert_eq!(base.absolute(96).unwrap(), 608);
        assert_eq!(base.relative(608).unwrap(), 96);
    }

    /// Both operands are file-derived, so a malformed file can overflow the sum.
    /// It is reported rather than wrapped into a valid-looking address.
    #[test]
    fn an_overflowing_sum_is_reported_not_wrapped() {
        let base = BaseAddress::new(512);
        assert_eq!(
            base.absolute(u64::MAX),
            Err(FormatError::OffsetOverflow {
                offset: u64::MAX,
                length: 512,
            })
        );
    }

    /// An address inside the userblock has no stored form: no HDF5 structure
    /// lives below the base. The two subtractions this replaced would have
    /// wrapped to a near-`u64::MAX` address instead.
    #[test]
    fn an_address_below_the_base_is_refused_rather_than_wrapped() {
        let base = BaseAddress::new(512);
        assert_eq!(
            base.relative(511),
            Err(FormatError::AddressBelowBase {
                address: 511,
                base: 512,
            })
        );
        // The boundary itself is the image's first byte, and does have one.
        assert_eq!(base.relative(512).unwrap(), 0);
    }
}
