//! Complex number newtypes recognized by the MAT serializer.
//!
//! These use a sentinel struct name so the generic MAT serializer can detect
//! complex values and write them as HDF5 compound `{real, imag}` datasets,
//! which is how MATLAB stores complex arrays in v7.3 files.
//!
//! For `Vec<Complex64>` the serializer produces a compound dataset of shape
//! `[1, N]`; a bare `Complex64` becomes a compound scalar of shape `[1, 1]`.
//!
//! There is one type per component class, because the component class *is* the
//! array's MATLAB class: a [`ComplexI16`] array reports `int16`, not some
//! complex-specific class, and a capture that samples as pairs of 16-bit
//! integers stores in half the space its `single` equivalent would take.
//! Nothing is ever widened on the way in or out — a value written as
//! [`ComplexI16`] reads back only as [`ComplexI16`], so the file always says
//! what it holds.
//!
//! For a large array, reach for the [`i16_array`]-style helpers below rather
//! than letting serde walk the elements: a sentinel struct per sample costs
//! about 26 ns, which is most of what writing a capture takes.

use core::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::mat::value::ComplexTag;

macro_rules! complex_types {
    ($($name:ident => $scalar:ty, $sentinel:ident, $array_sentinel:ident,
       $array_fn:ident, $tag:ident, $matlab:literal),* $(,)?) => {
        $(
            #[doc = concat!(
                "Sentinel struct name for [`", stringify!($name), "`]."
            )]
            pub(crate) const $sentinel: &str =
                concat!("__hdf5_pure_mat_", stringify!($name), "__");

            #[doc = concat!(
                "Sentinel newtype name for a whole slice of [`", stringify!($name),
                "`], written by [`", stringify!($array_fn), "`]."
            )]
            pub(crate) const $array_sentinel: &str =
                concat!("__hdf5_pure_mat_", stringify!($name), "_array__");

            #[doc = concat!(
                "A complex number with `", stringify!($scalar),
                "` components, stored as a MATLAB `", $matlab,
                "` complex array."
            )]
            ///
            /// `#[repr(C)]` so a slice of these is a valid input to the
            /// matching array helper, which reads the slice as raw bytes.
            #[derive(Debug, Clone, Copy, PartialEq)]
            #[repr(C)]
            pub struct $name {
                /// Real part.
                pub re: $scalar,
                /// Imaginary part.
                pub im: $scalar,
            }

            impl $name {
                /// Build a new complex value.
                pub const fn new(re: $scalar, im: $scalar) -> Self {
                    Self { re, im }
                }
            }

            impl Serialize for $name {
                fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                    let mut s = serializer.serialize_struct($sentinel, 2)?;
                    s.serialize_field("real", &self.re)?;
                    s.serialize_field("imag", &self.im)?;
                    s.end()
                }
            }

            impl<'de> Deserialize<'de> for $name {
                fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                    struct ComplexVisitor;

                    impl<'de> Visitor<'de> for ComplexVisitor {
                        type Value = $name;
                        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                            f.write_str(concat!(
                                stringify!($name),
                                " struct with fields `real` and `imag`"
                            ))
                        }
                        fn visit_map<A: MapAccess<'de>>(
                            self,
                            mut map: A,
                        ) -> Result<$name, A::Error> {
                            let mut re: Option<$scalar> = None;
                            let mut im: Option<$scalar> = None;
                            while let Some(key) = map.next_key::<String>()? {
                                match key.as_str() {
                                    "real" => re = Some(map.next_value()?),
                                    "imag" => im = Some(map.next_value()?),
                                    _ => {
                                        let _: serde::de::IgnoredAny = map.next_value()?;
                                    }
                                }
                            }
                            Ok($name {
                                re: re.ok_or_else(|| de::Error::missing_field("real"))?,
                                im: im.ok_or_else(|| de::Error::missing_field("imag"))?,
                            })
                        }
                    }

                    deserializer.deserialize_struct(
                        $sentinel,
                        &["real", "imag"],
                        ComplexVisitor,
                    )
                }
            }

            #[doc = concat!(
                "Serialize a slice of complex `", stringify!($scalar),
                "` as one MATLAB `", $matlab, "` complex array."
            )]
            ///
            /// A `serialize_with` helper. Serde has no bulk channel for a
            /// sequence, so the default `Vec<T>` path pays a serializer
            /// dispatch per sample; this hands the slice over whole and is the
            /// difference between a 0.15 GB/s write and a multi-GB/s one.
            ///
            #[doc = concat!(
                "Takes any [`ComplexElement`] whose `Component` is `",
                stringify!($scalar), "` — this module's [`", stringify!($name),
                "`], `num_complex::Complex<", stringify!($scalar),
                ">` under the `num-complex` feature, or your own type. The \
                 component is part of the bound, so a same-width class cannot \
                 slip through: `i32` parts will not compile here."
            )]
            ///
            /// Two things to know before annotating an existing field:
            ///
            /// - **The output is MAT-specific.** The slice crosses serde as a
            ///   byte string, so serializing the same struct to JSON or another
            ///   format yields raw bytes rather than a sequence, and its own
            ///   `Deserialize` will not read that back. Annotate a field only
            ///   on a type written to `.mat` alone.
            #[doc = concat!(
                "- **An empty slice keeps its component class**, writing an \
                 empty `", $matlab, "` array where a plain `Vec<",
                stringify!($name), ">` writes an empty `double`. That is the \
                 only case where the annotation changes the file."
            )]
            ///
            /// Reading is untouched: the file deserializes through the ordinary
            /// per-element path, which is now the slower half of a round trip.
            ///
            /// One-dimensional only — a [`Matrix`](crate::mat::Matrix) of
            /// complex values has no bulk path.
            ///
            /// # Example
            ///
            /// ```
            /// # use hdf5_pure::mat;
            /// # use serde::Serialize;
            /// #[derive(Serialize)]
            /// struct Capture {
            #[doc = concat!("    #[serde(serialize_with = \"mat::complex::",
                            stringify!($array_fn), "\")]")]
            #[doc = concat!("    samples: Vec<mat::", stringify!($name), ">,")]
            /// }
            /// ```
            pub fn $array_fn<S, T>(data: &[T], serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
                T: ComplexElement<Component = $scalar>,
            {
                // A backstop against a wrong `unsafe impl`, which is the only
                // way to reach this with a `T` of the wrong shape. Everything
                // else the byte view needs — initialized, no padding, real
                // first — is what the impl promised and cannot be checked.
                assert_eq!(
                    core::mem::size_of::<T>(),
                    2 * core::mem::size_of::<$scalar>(),
                    concat!(
                        "mat::complex::", stringify!($array_fn),
                        ": ComplexElement impl has the wrong element size",
                    ),
                );
                // SAFETY: `T: ComplexElement<Component = $scalar>` is an
                // unsafe promise that every byte of `T` is initialized data
                // laid out as two `$scalar`, so the whole slice is readable as
                // bytes; the size assert above catches an impl that got the
                // width wrong. Reading as `u8` needs no alignment, and the
                // length comes from the slice itself. Empty slices are fine:
                // `as_ptr` is non-null and aligned even when dangling.
                let raw: &[u8] = unsafe {
                    core::slice::from_raw_parts(
                        data.as_ptr().cast::<u8>(),
                        core::mem::size_of_val(data),
                    )
                };
                serializer.serialize_newtype_struct($array_sentinel, &RawComplexBytes(raw))
            }

            // SAFETY: declared `#[repr(C)]` above with exactly two `$scalar`
            // fields, real first. Two equal-width scalars leave no padding.
            unsafe impl ComplexElement for $name {
                type Component = $scalar;
            }

            // SAFETY: `num_complex::Complex<T>` is `#[repr(C)]` with `re`
            // then `im`, the same shape as `$name`.
            #[cfg(feature = "num-complex")]
            unsafe impl ComplexElement for num_complex::Complex<$scalar> {
                type Component = $scalar;
            }
        )*

        /// The component class a complex sentinel names, or `None` for any
        /// other struct name.
        ///
        /// This is how the (de)serializer recognizes a complex value: the
        /// sentinel carries the component class through serde, which has no
        /// way to pass the concrete Rust type along with the struct.
        pub(crate) fn complex_tag_for_sentinel(name: &str) -> Option<ComplexTag> {
            match name {
                $($sentinel => Some(ComplexTag::$tag),)*
                _ => None,
            }
        }

        /// The component class a bulk-array sentinel names, or `None` for any
        /// other newtype name. The slice counterpart of
        /// [`complex_tag_for_sentinel`].
        pub(crate) fn complex_tag_for_array_sentinel(name: &str) -> Option<ComplexTag> {
            match name {
                $($array_sentinel => Some(ComplexTag::$tag),)*
                _ => None,
            }
        }
    };
}

/// A type the array helpers may read as raw bytes: two `Component` fields,
/// real first, and nothing else.
///
/// Implemented for this module's [`ComplexI16`] and friends, and for
/// `num_complex::Complex<T>` under the `num-complex` feature. Implement it for
/// your own complex type to pass a slice of it to [`i16_array`] and company;
/// the orphan rule means a type from a third crate needs an impl from one of
/// the two crates that own it.
///
/// # Safety
///
/// `Self` must have a guaranteed layout (`#[repr(C)]` or `#[repr(transparent)]`)
/// of exactly two `Component` fields, real first, and every byte of it must be
/// initialized and meaningful as data. In particular no padding, no
/// `MaybeUninit`, no pointers or references, and no interior mutability: the
/// helpers copy `size_of::<Self>()` bytes per element straight into the file,
/// so a padding byte is both undefined behavior and an uninitialized byte
/// written to disk.
///
/// `Component` names the MATLAB class the array is stored as, which is what
/// stops a same-width mix-up — `i32` components cannot reach [`f32_array`]
/// even though the two elements are the same size and alignment.
pub unsafe trait ComplexElement: Copy {
    /// The scalar class of both parts.
    type Component;
}

/// The caller's slice as borrowed native-endian bytes. `serialize_bytes` is
/// serde's only bulk channel, so a serializer that does not know the sentinel
/// sees a plain byte string — which is why these helpers are MAT-specific.
struct RawComplexBytes<'a>(&'a [u8]);

impl Serialize for RawComplexBytes<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(self.0)
    }
}

complex_types! {
    Complex64 => f64, COMPLEX64_SENTINEL, COMPLEX64_ARRAY_SENTINEL, f64_array, F64, "double",
    Complex32 => f32, COMPLEX32_SENTINEL, COMPLEX32_ARRAY_SENTINEL, f32_array, F32, "single",
    ComplexI64 => i64, COMPLEX_I64_SENTINEL, COMPLEX_I64_ARRAY_SENTINEL, i64_array, I64, "int64",
    ComplexI32 => i32, COMPLEX_I32_SENTINEL, COMPLEX_I32_ARRAY_SENTINEL, i32_array, I32, "int32",
    ComplexI16 => i16, COMPLEX_I16_SENTINEL, COMPLEX_I16_ARRAY_SENTINEL, i16_array, I16, "int16",
    ComplexI8 => i8, COMPLEX_I8_SENTINEL, COMPLEX_I8_ARRAY_SENTINEL, i8_array, I8, "int8",
    ComplexU64 => u64, COMPLEX_U64_SENTINEL, COMPLEX_U64_ARRAY_SENTINEL, u64_array, U64, "uint64",
    ComplexU32 => u32, COMPLEX_U32_SENTINEL, COMPLEX_U32_ARRAY_SENTINEL, u32_array, U32, "uint32",
    ComplexU16 => u16, COMPLEX_U16_SENTINEL, COMPLEX_U16_ARRAY_SENTINEL, u16_array, U16, "uint16",
    ComplexU8 => u8, COMPLEX_U8_SENTINEL, COMPLEX_U8_ARRAY_SENTINEL, u8_array, U8, "uint8",
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_compare() {
        let a = Complex64::new(1.0, -2.0);
        let b = Complex64 { re: 1.0, im: -2.0 };
        assert_eq!(a, b);
    }

    #[test]
    fn each_sentinel_names_its_own_component_class() {
        assert_eq!(
            complex_tag_for_sentinel(COMPLEX_I16_SENTINEL),
            Some(ComplexTag::I16)
        );
        assert_eq!(
            complex_tag_for_sentinel(COMPLEX64_SENTINEL),
            Some(ComplexTag::F64)
        );
        assert_eq!(complex_tag_for_sentinel("SomeUserStruct"), None);
    }

    /// Drive the byte view the array helpers take, small enough for Miri to
    /// interpret — the CI Miri job runs this module, and the unsafe here is
    /// the reason it does.
    ///
    /// The interesting inputs are the ones a size-and-alignment check could
    /// not have rejected: an empty slice (whose `as_ptr` is dangling) and a
    /// one-element slice, both of which have to produce the same value the
    /// element-wise path would.
    #[test]
    fn the_array_helpers_read_their_slice_without_undefined_behavior() {
        use crate::mat::options::Options;
        use crate::mat::ser::value_ser::ValueSerializer;
        use crate::mat::value::{ComplexVec, MatValue};

        let opts = Options::default();

        let pairs = [ComplexI16::new(1, -2), ComplexI16::new(3, -4)];
        assert_eq!(
            i16_array(&pairs, ValueSerializer::new(&opts)).unwrap(),
            MatValue::ComplexVec1D(ComplexVec::I16(vec![(1, -2), (3, -4)])),
        );

        let none: &[ComplexI16] = &[];
        assert_eq!(
            i16_array(none, ValueSerializer::new(&opts)).unwrap(),
            MatValue::ComplexVec1D(ComplexVec::I16(Vec::new())),
        );

        // A wider component, so a stride error shows up as a wrong value
        // rather than as a coincidence.
        let wide = [Complex64::new(1.5, -2.5)];
        assert_eq!(
            f64_array(&wide, ValueSerializer::new(&opts)).unwrap(),
            MatValue::ComplexVec1D(ComplexVec::F64(vec![(1.5, -2.5)])),
        );
    }

    /// The sentinel is the only thing distinguishing one complex type from
    /// another once serde has erased the Rust type, so two sharing a name
    /// would silently write each other's class.
    #[test]
    fn sentinels_are_distinct() {
        let all = [
            COMPLEX64_SENTINEL,
            COMPLEX32_SENTINEL,
            COMPLEX_I64_SENTINEL,
            COMPLEX_I32_SENTINEL,
            COMPLEX_I16_SENTINEL,
            COMPLEX_I8_SENTINEL,
            COMPLEX_U64_SENTINEL,
            COMPLEX_U32_SENTINEL,
            COMPLEX_U16_SENTINEL,
            COMPLEX_U8_SENTINEL,
        ];
        let unique: std::collections::HashSet<&str> = all.iter().copied().collect();
        assert_eq!(unique.len(), all.len());
    }
}
