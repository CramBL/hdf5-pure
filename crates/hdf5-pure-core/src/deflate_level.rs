use crate::FormatError;

/// A Deflate compression level in `0..=9`.
///
/// Numeric conversions from [`u8`] or [`u32`] return [`FormatError::InvalidDeflateLevel`] for
/// values above `9`. The C library sets this level with `H5Pset_deflate`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeflateLevel {
    /// Compression level `0`, with zlib framing and no compression.
    Zero,
    /// Compression level `1`, optimized for speed.
    One,
    /// Compression level `2`.
    Two,
    /// Compression level `3`.
    Three,
    /// Compression level `4`.
    Four,
    /// Compression level `5`.
    Five,
    /// Compression level `6`.
    Six,
    /// Compression level `7`.
    Seven,
    /// Compression level `8`.
    Eight,
    /// Compression level `9`, optimized for compressed size.
    Nine,
}

impl DeflateLevel {
    /// Returns the numeric compression level.
    pub const fn get(self) -> u8 {
        match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::Two => 2,
            Self::Three => 3,
            Self::Four => 4,
            Self::Five => 5,
            Self::Six => 6,
            Self::Seven => 7,
            Self::Eight => 8,
            Self::Nine => 9,
        }
    }
}

impl TryFrom<u8> for DeflateLevel {
    type Error = FormatError;

    /// Converts a numeric compression level in `0..=9`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDeflateLevel`] if `level` exceeds `9`.
    fn try_from(level: u8) -> Result<Self, FormatError> {
        Self::try_from(u32::from(level))
    }
}

impl TryFrom<u32> for DeflateLevel {
    type Error = FormatError;

    /// Converts a numeric compression level in `0..=9`.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidDeflateLevel`] if `level` exceeds `9`.
    fn try_from(level: u32) -> Result<Self, FormatError> {
        Ok(match level {
            0 => Self::Zero,
            1 => Self::One,
            2 => Self::Two,
            3 => Self::Three,
            4 => Self::Four,
            5 => Self::Five,
            6 => Self::Six,
            7 => Self::Seven,
            8 => Self::Eight,
            9 => Self::Nine,
            _ => return Err(FormatError::InvalidDeflateLevel { level }),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn parses_every_byte(
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] high: u8,
        #[values(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)] low: u8,
    ) {
        let raw = high * 16 + low;
        let levels = [
            DeflateLevel::Zero,
            DeflateLevel::One,
            DeflateLevel::Two,
            DeflateLevel::Three,
            DeflateLevel::Four,
            DeflateLevel::Five,
            DeflateLevel::Six,
            DeflateLevel::Seven,
            DeflateLevel::Eight,
            DeflateLevel::Nine,
        ];
        if let Some(&level) = levels.get(usize::from(raw)) {
            assert_eq!(DeflateLevel::try_from(raw).unwrap(), level);
            assert_eq!(DeflateLevel::try_from(u32::from(raw)).unwrap(), level);
            assert_eq!(level.get(), raw);
        } else {
            assert_eq!(
                DeflateLevel::try_from(raw).unwrap_err(),
                FormatError::InvalidDeflateLevel {
                    level: u32::from(raw)
                }
            );
        }
    }

    #[rstest]
    #[case(10)]
    #[case(u32::from(u8::MAX))]
    #[case(256)]
    #[case(262)]
    #[case(u32::MAX)]
    fn rejects_invalid_metadata(#[case] level: u32) {
        assert_eq!(
            DeflateLevel::try_from(level).unwrap_err(),
            FormatError::InvalidDeflateLevel { level }
        );
    }
}
