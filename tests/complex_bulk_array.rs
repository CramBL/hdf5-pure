#![cfg(feature = "serde")]
//! The bulk complex-array helpers in [`mat::complex`].
//!
//! Serde has no bulk channel for a sequence, so a `Vec<ComplexI16>` pays a
//! serializer dispatch per sample — about 26 ns, which for a capture-sized
//! array is the whole cost of writing the file. The helpers hand the slice
//! over whole instead.
//!
//! That makes byte-for-byte equivalence with the element-wise path the
//! property worth pinning: the helpers exist to change how long a write takes,
//! not what it produces. The one deliberate exception is the empty slice, and
//! it has its own test below.

use hdf5_pure::mat::options::OneDimensionalMode;
use hdf5_pure::mat::{
    self, Complex32, Complex64, ComplexElement, ComplexI8, ComplexI16, ComplexI32, ComplexI64,
    ComplexU8, ComplexU16, ComplexU32, ComplexU64, Compression, Options,
};
use hdf5_pure::{AttrValue, File, LibVer};
use serde::{Deserialize, Serialize, Serializer};

/// A single pair, odd and even counts, and an element count past `u16::MAX`.
/// The empty slice is deliberately absent — it is the one length where the two
/// paths differ, so it has tests of its own below.
const LENGTHS: &[usize] = &[1, 2, 3, 7, 64, 1000, 65537];

/// One equivalence test per component class: the same values written both
/// ways, compared as whole files.
///
/// Per class rather than once generically, because the class is what the
/// helper selects — a helper wired to the wrong width or the wrong sentinel
/// would still round-trip through this crate and still be wrong in MATLAB.
macro_rules! equivalence_per_class {
    ($($test:ident => $elem:ident, $scalar:ty, $helper:path),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                #[derive(Serialize)]
                struct Elementwise {
                    data: Vec<$elem>,
                }

                struct Slice<'a>(&'a [$elem]);

                impl Serialize for Slice<'_> {
                    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                        $helper(self.0, s)
                    }
                }

                #[derive(Serialize)]
                struct Bulk<'a> {
                    data: Slice<'a>,
                }

                for &n in LENGTHS {
                    let data: Vec<$elem> = (0..n)
                        .map(|i| <$elem>::new(i as $scalar, (n - i) as $scalar))
                        .collect();
                    let element_wise = mat::to_bytes(&Elementwise { data: data.clone() })
                        .expect("element-wise write");
                    let bulk = mat::to_bytes(&Bulk { data: Slice(&data) }).expect("bulk write");
                    assert_eq!(
                        element_wise,
                        bulk,
                        concat!(stringify!($elem), ": bulk and element-wise files differ at {} samples"),
                        n,
                    );
                }
            }
        )*
    };
}

equivalence_per_class! {
    f64_array_matches_element_wise => Complex64, f64, mat::complex::f64_array,
    f32_array_matches_element_wise => Complex32, f32, mat::complex::f32_array,
    i64_array_matches_element_wise => ComplexI64, i64, mat::complex::i64_array,
    i32_array_matches_element_wise => ComplexI32, i32, mat::complex::i32_array,
    i16_array_matches_element_wise => ComplexI16, i16, mat::complex::i16_array,
    i8_array_matches_element_wise => ComplexI8, i8, mat::complex::i8_array,
    u64_array_matches_element_wise => ComplexU64, u64, mat::complex::u64_array,
    u32_array_matches_element_wise => ComplexU32, u32, mat::complex::u32_array,
    u16_array_matches_element_wise => ComplexU16, u16, mat::complex::u16_array,
    u8_array_matches_element_wise => ComplexU8, u8, mat::complex::u8_array,
}

// ---------------------------------------------------------------------------
// The documented usage: a `serialize_with` attribute on a struct field
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Capture {
    label: String,
    #[serde(serialize_with = "mat::complex::i16_array")]
    samples: Vec<ComplexI16>,
}

/// The bulk path is a write-side optimization only, so the file it produces
/// has to deserialize through the ordinary `Vec<ComplexI16>` reader.
#[test]
fn a_bulk_written_array_reads_back_element_wise() {
    let capture = Capture {
        label: "sweep".to_owned(),
        samples: (0..1000).map(|i| ComplexI16::new(i, -i)).collect(),
    };
    let bytes = mat::to_bytes(&capture).expect("write");
    let back: Capture = mat::from_bytes(&bytes).expect("read");
    assert_eq!(back, capture);
}

/// The helper lowers to the same value tree the element-wise path builds, so
/// every option still reaches the writer that emits it. The equivalence tests
/// above run under the defaults only, which would not notice a bulk path that
/// had grown a writer of its own and quietly stopped compressing or
/// reorienting.
#[test]
fn options_reach_a_bulk_array_the_same_way() {
    #[derive(Serialize)]
    struct Elementwise {
        samples: Vec<ComplexI16>,
    }

    // A repeating ramp, so a working deflate has something to find.
    let samples: Vec<ComplexI16> = (0..4096)
        .map(|i| ComplexI16::new(i % 8, -(i % 8)))
        .collect();

    // Compression needs chunked storage, whose chunk indices need the 1.10
    // format; the MAT default is 1.8, which refuses compression by name.
    let mut options = Options::default();
    options.libver = LibVer::V110;
    options.compression = Compression::Deflate {
        level: 6,
        shuffle: false,
    };
    options.one_dimensional_mode = OneDimensionalMode::RowVector;

    let element_wise = mat::to_bytes_with_options(
        &Elementwise {
            samples: samples.clone(),
        },
        &options,
    )
    .expect("element-wise");
    let bulk = mat::to_bytes_with_options(
        &Capture {
            label: "ramp".to_owned(),
            samples: samples.clone(),
        },
        &options,
    )
    .expect("bulk");

    // The bulk file carries an extra `label` variable, so compare the sample
    // dataset rather than the whole file. Shape and class as well as payload:
    // the payload alone is blind to orientation, so a bulk path that built a
    // `ComplexMatrix` instead of a `ComplexVec1D` would pass every other test
    // here and silently transpose the field under `RowVector`.
    assert_eq!(
        payload_of(&element_wise, "samples"),
        payload_of(&bulk, "samples"),
    );
    assert_eq!(dims_of(&element_wise, "samples"), dims_of(&bulk, "samples"));
    assert_eq!(
        class_of(&element_wise, "samples"),
        class_of(&bulk, "samples")
    );

    // And the options really were in force, rather than both paths having
    // ignored them identically.
    let mut plain_opts = options.clone();
    plain_opts.compression = Compression::None;
    plain_opts.one_dimensional_mode = OneDimensionalMode::ColumnVector;
    let plain = mat::to_bytes_with_options(
        &Capture {
            label: "ramp".to_owned(),
            samples,
        },
        &plain_opts,
    )
    .expect("uncompressed bulk");
    assert!(
        bulk.len() < plain.len(),
        "deflate should shrink a repeating ramp: {} vs {}",
        bulk.len(),
        plain.len()
    );
    assert_ne!(
        dims_of(&bulk, "samples"),
        dims_of(&plain, "samples"),
        "the orientation option has to move the shape, or the shape check above proves nothing"
    );
}

/// Every float bit pattern MATLAB distinguishes has to survive the byte view.
///
/// The equivalence tests above generate their samples with `i as f64`, so they
/// are all finite, non-negative and integral — a decode that normalized `-0.0`
/// to `0.0`, or quieted a signaling NaN, would pass every one of them. MATLAB
/// tells `-0` from `0` (`1/x` is `-Inf`), so it is a real distinction to lose.
#[test]
fn float_specials_survive_the_bulk_path() {
    #[derive(Serialize)]
    struct Elementwise {
        data: Vec<Complex64>,
    }

    struct Slice<'a>(&'a [Complex64]);

    impl Serialize for Slice<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            mat::complex::f64_array(self.0, s)
        }
    }

    #[derive(Serialize)]
    struct Bulk<'a> {
        data: Slice<'a>,
    }

    let values = vec![
        Complex64::new(f64::NAN, -f64::NAN),
        Complex64::new(f64::INFINITY, f64::NEG_INFINITY),
        Complex64::new(-0.0, 0.0),
        Complex64::new(f64::MIN_POSITIVE / 2.0, -1.5),
    ];

    assert_eq!(
        mat::to_bytes(&Elementwise {
            data: values.clone()
        })
        .expect("element-wise"),
        mat::to_bytes(&Bulk {
            data: Slice(&values)
        })
        .expect("bulk"),
    );
}

/// A dataset's element bytes, filters already undone by the reader.
fn payload_of(bytes: &[u8], name: &str) -> Vec<u8> {
    let file = File::from_bytes(bytes.to_vec()).expect("open");
    file.dataset(name)
        .expect("dataset")
        .read_u8()
        .expect("payload")
}

/// The helper reads its payload out of live memory, so the bytes it starts
/// from are the *host's* order — and HDF5 stores little-endian. This asserts
/// the components as they land on disk, which holds on either kind of host and
/// is the assertion a big-endian one would fail if the swap were ever dropped.
///
/// The equivalence tests above compare two paths against each other, so they
/// would agree about a wrong byte order; this one names it.
///
/// It still cannot fail on a little-endian host, where the native and
/// little-endian decoders are the same function, and CI has no big-endian
/// target. What it does pin is the LE encode on the way out.
#[test]
fn a_bulk_array_stores_its_components_little_endian() {
    struct Slice<'a>(&'a [ComplexI32]);

    impl Serialize for Slice<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            mat::complex::i32_array(self.0, s)
        }
    }

    #[derive(Serialize)]
    struct Root<'a> {
        data: Slice<'a>,
    }

    let bytes = mat::to_bytes(&Root {
        data: Slice(&[ComplexI32::new(1, 2)]),
    })
    .expect("write");

    assert_eq!(payload_of(&bytes, "data"), vec![1, 0, 0, 0, 2, 0, 0, 0]);
}

// ---------------------------------------------------------------------------
// The empty slice: the one place the two paths deliberately differ
// ---------------------------------------------------------------------------

/// An empty `Vec<ComplexI16>` reveals no element, so the element-wise path
/// cannot recover the component class and falls back to an empty `double`.
/// The helper was told the class by name, so it keeps it — matching
/// `MatBuilder::write_complex_i16`, which documents the same answer for an
/// empty input, and the `Matrix<Complex*>` sentinels, which exist for exactly
/// this problem.
///
/// This is the only case where switching a field to a helper changes the file.
#[test]
fn an_empty_bulk_array_keeps_its_component_class_where_the_element_wise_path_cannot() {
    #[derive(Serialize, Deserialize)]
    struct Elementwise {
        data: Vec<ComplexI16>,
    }

    let bulk = mat::to_bytes(&Capture {
        label: "none".to_owned(),
        samples: Vec::new(),
    })
    .expect("bulk write");
    let element_wise = mat::to_bytes(&Elementwise { data: Vec::new() }).expect("element-wise");

    assert_ne!(
        class_of(&bulk, "samples"),
        class_of(&element_wise, "data"),
        "the empty-slice difference this test documents has gone away"
    );
    assert_eq!(class_of(&bulk, "samples"), "int16");
    assert_eq!(class_of(&element_wise, "data"), "double");

    // Both read back as an empty array, which is what makes the difference a
    // matter of what the file says it holds rather than of what it holds.
    let from_bulk: Capture = mat::from_bytes(&bulk).expect("read bulk");
    assert!(from_bulk.samples.is_empty());
    let from_element_wise: Elementwise = mat::from_bytes(&element_wise).expect("read element-wise");
    assert!(from_element_wise.data.is_empty());
}

/// The two MAT emitters have to describe an empty bulk array with the same
/// MATLAB dimensions.
///
/// They already drifted once over an empty `Vec` (`0x0` against `0x1`, see the
/// note on `mat::dims::vector_dims`), and this feature reopened the same hole:
/// an empty `ComplexVec1D` was unreachable from the serializer until these
/// helpers existed, so the default emitter's hardcoded `[1, n]` had never been
/// asked for `n == 0`. `serde_roundtrip::both_emit_paths_produce_the_same_bytes`
/// carries the general parity check; this one also pins the shape, because a
/// comparison passes just as well when both sides are wrong.
#[test]
fn an_empty_bulk_array_has_the_same_dims_from_either_emitter() {
    let empty = Capture {
        label: "none".to_owned(),
        samples: Vec::new(),
    };
    let default_path = mat::to_bytes(&empty).expect("to_bytes");
    let options_path =
        mat::to_bytes_with_options(&empty, &Options::default()).expect("to_bytes_with_options");

    assert_eq!(
        dims_of(&default_path, "samples"),
        dims_of(&options_path, "samples"),
        "the two emitters disagree about an empty complex array's shape"
    );
    // MATLAB's own `[]`, which is the shape every other empty here uses.
    assert_eq!(dims_of(&default_path, "samples"), vec![0, 0]);
    assert_eq!(default_path, options_path);
}

/// The whole point of the `num-complex` feature: a slice of the de-facto
/// standard Rust complex type reaches the helper directly.
///
/// The orphan rule means this impl can only ship from here, so nothing outside
/// this crate can test it — and without a test, a `num-complex` major that
/// changed `Complex`'s layout or field order would go unnoticed until it
/// silently swapped every sample's parts.
#[cfg(feature = "num-complex")]
#[test]
fn a_num_complex_slice_writes_the_same_file() {
    #[derive(Serialize)]
    struct Elementwise {
        data: Vec<ComplexI16>,
    }

    struct Foreign<'a>(&'a [num_complex::Complex<i16>]);

    impl Serialize for Foreign<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            mat::complex::i16_array(self.0, s)
        }
    }

    #[derive(Serialize)]
    struct Bulk<'a> {
        data: Foreign<'a>,
    }

    // Distinct, sign-varying parts, so a swap or a reinterpretation shows.
    let foreign: Vec<num_complex::Complex<i16>> = (0..500)
        .map(|i| num_complex::Complex::new(i, -i - 1))
        .collect();
    let native: Vec<ComplexI16> = foreign
        .iter()
        .map(|c| ComplexI16::new(c.re, c.im))
        .collect();

    assert_eq!(
        mat::to_bytes(&Elementwise { data: native }).expect("element-wise"),
        mat::to_bytes(&Bulk {
            data: Foreign(&foreign)
        })
        .expect("bulk"),
    );
}

/// A dataset's HDF5 shape.
fn dims_of(bytes: &[u8], name: &str) -> Vec<u64> {
    let file = File::from_bytes(bytes.to_vec()).expect("open");
    file.dataset(name)
        .expect("dataset")
        .shape()
        .expect("shape")
        .to_vec()
}

/// The `MATLAB_class` attribute of a top-level variable.
fn class_of(bytes: &[u8], name: &str) -> String {
    let file = File::from_bytes(bytes.to_vec()).expect("open");
    let ds = file.dataset(name).expect("dataset");
    ds.attrs()
        .expect("attrs")
        .get("MATLAB_class")
        .and_then(AttrValue::as_str)
        .unwrap_or_else(|| panic!("{name} has no MATLAB_class"))
        .to_owned()
}

// ---------------------------------------------------------------------------
// Foreign element types
// ---------------------------------------------------------------------------

/// A complex type this crate does not own — `num_complex::Complex<i16>`, an
/// FFI `struct { re, im }` — is why the helpers take a [`ComplexElement`]
/// rather than `&[ComplexI16]`. Requiring a conversion pass would reintroduce
/// the per-sample cost they exist to remove.
#[test]
fn a_foreign_element_writes_the_same_file() {
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct ForeignIq {
        re: i16,
        im: i16,
    }

    // SAFETY: `#[repr(C)]` with exactly two `i16`, real first, so no padding
    // and every byte is data.
    unsafe impl ComplexElement for ForeignIq {
        type Component = i16;
    }

    #[derive(Serialize)]
    struct Elementwise {
        data: Vec<ComplexI16>,
    }

    struct Foreign<'a>(&'a [ForeignIq]);

    impl Serialize for Foreign<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            mat::complex::i16_array(self.0, s)
        }
    }

    #[derive(Serialize)]
    struct Bulk<'a> {
        data: Foreign<'a>,
    }

    let foreign: Vec<ForeignIq> = (0..500).map(|i| ForeignIq { re: i, im: !i }).collect();
    let native: Vec<ComplexI16> = foreign
        .iter()
        .map(|p| ComplexI16::new(p.re, p.im))
        .collect();

    assert_eq!(
        mat::to_bytes(&Elementwise { data: native }).expect("element-wise"),
        mat::to_bytes(&Bulk {
            data: Foreign(&foreign)
        })
        .expect("bulk"),
    );
}

/// The helpers read `2 * size_of::<Component>()` bytes per element, so an
/// `unsafe impl` that named a wider type would read past the slice. Nothing
/// safe can reach this — a plain `&[i16]` or `&[u32]` is now a compile error,
/// and so is `i32` components handed to `f32_array` — but a wrong impl still
/// gets a diagnostic rather than a buffer overrun.
#[test]
#[should_panic(expected = "ComplexElement impl has the wrong element size")]
fn an_impl_that_lies_about_its_width_is_caught() {
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct TooWide {
        re: i32,
        im: i32,
    }

    // SAFETY: deliberately wrong, to exercise the backstop. `Component = i16`
    // claims a 4-byte element where this one is 8.
    unsafe impl ComplexElement for TooWide {
        type Component = i16;
    }

    struct Wrong<'a>(&'a [TooWide]);

    impl Serialize for Wrong<'_> {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            mat::complex::i16_array(self.0, s)
        }
    }

    #[derive(Serialize)]
    struct Root<'a> {
        data: Wrong<'a>,
    }

    let _ = mat::to_bytes(&Root {
        data: Wrong(&[TooWide { re: 1, im: 2 }]),
    });
}
