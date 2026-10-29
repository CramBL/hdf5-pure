use crate::FormatError;

/// A finite, positive ZFP compression rate of at most 64 bits per scalar.
///
/// The dataset's scalar width and chunk geometry impose further limits. A rate of 64 is
/// valid for 64-bit scalars. The writer rejects a block budget below one bit when the chunk
/// dimensions are known. Floating-point block headers may require more bits,
/// depending on the input values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZfpRate(f64);

impl ZfpRate {
    /// Returns the compression rate in bits per scalar.
    pub const fn get(self) -> f64 {
        self.0
    }

    /// The maximum requested rate, 64 compressed bits per scalar.
    pub const MAX: Self = Self(64.0);
}

impl TryFrom<f64> for ZfpRate {
    type Error = FormatError;

    /// Parses a requested number of compressed bits per scalar.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::InvalidZfpRate`] if `rate` is not finite, is at most zero, or
    /// exceeds 64.
    fn try_from(rate: f64) -> Result<Self, FormatError> {
        if rate.is_finite() && rate > 0.0 && rate <= Self::MAX.get() {
            Ok(Self(rate))
        } else {
            Err(FormatError::InvalidZfpRate {
                rate_bits: rate.to_bits(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::sixteen(16.0)]
    #[case::thirty_two(32.0)]
    #[case::maximum(64.0)]
    #[case::smallest_normal(f64::MIN_POSITIVE)]
    #[case::smallest_subnormal(f64::from_bits(1))]
    fn parses_finite_positive_rates_up_to_sixty_four(#[case] rate: f64) {
        assert_eq!(ZfpRate::try_from(rate).unwrap().get(), rate);
    }

    #[rstest]
    #[case::not_a_number(f64::NAN)]
    #[case::positive_infinity(f64::INFINITY)]
    #[case::negative_infinity(f64::NEG_INFINITY)]
    #[case::positive_zero(0.0)]
    #[case::negative_zero(-0.0)]
    #[case::negative_one(-1.0)]
    #[case::above_maximum(64.0001)]
    fn rejects_rates_outside_the_finite_positive_range(#[case] rate: f64) {
        assert_eq!(
            ZfpRate::try_from(rate).unwrap_err(),
            FormatError::InvalidZfpRate {
                rate_bits: rate.to_bits()
            }
        );
    }
}
