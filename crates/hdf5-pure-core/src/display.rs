//! Shares formatting for datatype, layout, and attribute summaries.
//!
//! Shapes use `2x3`, and truncated lists end in `… N more`.
//!
//! [`EscapedName`] is used where a name comes from the file.

use core::fmt;
use core::fmt::Write as _;

/// How many members a compound or an enumeration writes before eliding the
/// rest, in either type view.
///
/// A member list describes schema, so this limit exceeds the
/// eight elements an `AttrValue` writes of a value. It
/// still needs a bound: the member count is an on-disk `u16`, so one compound
/// can declare 65,535 members, each recursing into a datatype of its own, and
/// the whole of it would land in a single error message.
pub const DISPLAY_MAX_MEMBERS: usize = 16;

/// How many characters of one escaped name are written before it is truncated.
///
/// [`DISPLAY_MAX_MEMBERS`] bounds the number of names in a message. This bounds
/// how long each one is. Without it the two are not a bound at all: a name has
/// no length limit on disk either, so sixteen of them could still run to
/// megabytes.
const DISPLAY_MAX_NAME_CHARS: usize = 64;

/// Both caps exist to keep a hostile file's message readable, so their values
/// are held to that, not just the mechanism that applies them: a file can
/// declare 65,535 members of unbounded length, and the product of these two is
/// what a message is then worth. Raising either past what a person will read is
/// a compile error, not a silent loss of the bound.
const _: () = assert!(DISPLAY_MAX_MEMBERS <= 64);
const _: () = assert!(DISPLAY_MAX_NAME_CHARS <= 256);

/// A dimension list, as `4` or `2x3`.
pub struct Dims<'a, T>(pub &'a [T]);

impl<T: fmt::Display> fmt::Display for Dims<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str("x")?;
            }
            write!(f, "{dim}")?;
        }
        Ok(())
    }
}

/// Arbitrary file bytes, quoted, with anything outside printable ASCII escaped.
pub struct QuotedBytes<'a>(pub &'a [u8]);

impl fmt::Display for QuotedBytes<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("\"")?;
        for &byte in self.0 {
            if byte == b'"' || byte == b'\\' {
                write!(f, "\\{}", byte as char)?;
            } else if byte.is_ascii_graphic() || byte == b' ' {
                write!(f, "{}", byte as char)?;
            } else {
                write!(f, "\\x{byte:02x}")?;
            }
        }
        f.write_str("\"")
    }
}

/// A name read from the file, escaped so it cannot carry control characters
/// into a message, and truncated so it cannot flood one.
///
/// A member name, an enum label and a filter name are all decoded with
/// `String::from_utf8_lossy`, which rejects nothing a file can hold: a
/// crafted name can carry a newline, a NUL or a terminal escape sequence
/// straight into the message that quotes it. Delegates to
/// [`str::escape_debug`], so an ordinary name passes through unchanged and a
/// hostile one arrives as Rust escapes.
///
/// This bounds a name to one line of printable output. It does not make every
/// name unambiguous  -  one holding `, ` still reads like two, and one written in
/// a right-to-left script still reorders the text around it  -  so a caller that
/// needs to tell two names apart wants `Debug`.
pub struct EscapedName<'a>(pub &'a str);

impl fmt::Display for EscapedName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut escaped = self.0.escape_debug();
        for ch in escaped.by_ref().take(DISPLAY_MAX_NAME_CHARS) {
            f.write_char(ch)?;
        }
        if escaped.next().is_some() {
            f.write_str("…")?;
        }
        Ok(())
    }
}

/// The `, … N more` tail of a truncated list, written only when the list was in
/// fact truncated.
pub fn write_elided(f: &mut fmt::Formatter<'_>, elided: usize) -> fmt::Result {
    if elided > 0 {
        write!(f, ", … {elided} more")?;
    }
    Ok(())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    #[test]
    fn dims_joins_dimensions_with_x() {
        assert_eq!(Dims(&[4]).to_string(), "4");
        assert_eq!(Dims(&[2, 3]).to_string(), "2x3");
        assert_eq!(Dims::<u64>(&[]).to_string(), "");
    }

    /// The file decides these bytes, so a message quoting one stays on the line
    /// it was written on.
    #[test]
    fn an_escaped_name_cannot_carry_a_control_character() {
        assert_eq!(EscapedName("x").to_string(), "x");
        assert_eq!(EscapedName("température").to_string(), "température");

        for hostile in ["a\nb", "a\rb", "a\tb", "a\u{0}b", "a\u{1b}[31mb"] {
            let shown = EscapedName(hostile).to_string();
            assert!(
                !shown.chars().any(char::is_control),
                "{hostile:?} -> {shown}"
            );
        }
    }

    #[test]
    fn a_long_name_is_truncated() {
        let long = "n".repeat(DISPLAY_MAX_NAME_CHARS * 4);
        let shown = EscapedName(&long).to_string();

        assert_eq!(shown.chars().count(), DISPLAY_MAX_NAME_CHARS + 1);
        assert!(shown.ends_with('…'), "{shown}");

        let at_cap = "n".repeat(DISPLAY_MAX_NAME_CHARS);
        assert_eq!(EscapedName(&at_cap).to_string(), at_cap);
    }

    #[test]
    fn an_elided_tail_appears_only_when_something_was_dropped() {
        assert_eq!(fmt_elided(0), "");
        assert_eq!(fmt_elided(5), ", … 5 more");
    }

    fn fmt_elided(elided: usize) -> String {
        struct Tail(usize);
        impl fmt::Display for Tail {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write_elided(f, self.0)
            }
        }
        Tail(elided).to_string()
    }
}
