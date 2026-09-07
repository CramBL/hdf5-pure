// Crosschecks link the reference HDF5 C library (the `hdf5-metno` dev-dependency),
// gated to 64-bit little-endian targets; skip elsewhere so the pure-Rust suite still
// runs under `cross test --target i686-...`.
#![cfg(all(not(target_pointer_width = "32"), target_endian = "little"))]
#![cfg(feature = "__hdf5-1.10")]
//! Scale-offset encoding with a **defined fill value** (issue #287).
//!
//! The reference library sets `H5Z_SCALEOFFSET_PARM_FILAVAIL` to `FILL_DEFINED`
//! whenever the dataset carries a fill value, which is the shape libhdf5 and
//! h5py produce by default. In that mode an element equal to the fill value is
//! stored as the all-ones offset rather than as `value - min`, and the encoder
//! reserves that code point by widening `minbits` — both of which this crate's
//! encoder refused outright before this issue.
//!
//! Every test builds its fixture with the C library so the `cd_values` under
//! test are the reference library's own, then checks that what hdf5-pure
//! encodes is what the C library reads back.

use hdf5::Extent;
use hdf5::file::LibraryVersion;
use hdf5::filters::ScaleOffset as CScaleOffset;
use tempfile::tempdir;

/// Create a rank-1 unlimited scale-offset `u64` dataset with the C library,
/// under the latest format so it gets an Extensible-Array index (which is what
/// `append_staged` requires), carrying a defined fill value.
fn c_create(path: &std::path::Path, data: &[u64], chunk: usize, fill: u64) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let ds = file
        .new_dataset::<u64>()
        .scale_offset(CScaleOffset::Integer(0))
        .chunk((chunk,))
        .shape((Extent::resizable(data.len()),))
        .fill_value(fill)
        .create("col")
        .unwrap();
    ds.write_raw(data).unwrap();
    file.close().unwrap();
}

fn read_c(path: &std::path::Path) -> Vec<u64> {
    hdf5::File::open(path)
        .unwrap()
        .dataset("col")
        .unwrap()
        .read_raw::<u64>()
        .unwrap()
}

fn read_pure(path: &std::path::Path) -> Vec<u64> {
    hdf5_pure::File::open(path)
        .unwrap()
        .dataset("col")
        .unwrap()
        .read_u64()
        .unwrap()
}

fn pure_append(path: &std::path::Path, values: &[u64]) {
    let f = hdf5_pure::File::open_rw(path).unwrap();
    f.dataset("col")
        .unwrap()
        .append_staged(|b| {
            b.append_u64(values);
        })
        .unwrap();
    f.commit().unwrap();
}

/// The premise of the whole file, and wider than the issue that prompted it:
/// the reference library records `FILL_DEFINED` on **every** scale-offset
/// dataset it writes, not only those given a fill value. `H5Z__set_local`
/// consults `H5P_fill_value_defined`, which answers `UNDEFINED` only when a
/// caller explicitly sets the fill value to NULL — the untouched default
/// answers `DEFAULT`, which takes the same branch as a user-defined value.
///
/// So the refusal this file removes did not apply to the occasional dataset
/// with an explicit fill value. It applied to essentially every scale-offset
/// dataset written by libhdf5 or h5py.
#[test]
fn the_c_library_records_a_defined_fill_value_even_without_one() {
    let dir = tempdir().unwrap();
    let filavail_and_filval = |fill: Option<u64>, name: &str| {
        let path = dir.path().join(name);
        c_create_typed(&path, &[1u64, 2, 3], 512, CScaleOffset::Integer(0), fill);
        let f = hdf5_pure::File::open(&path).unwrap();
        let ds = f.dataset("col").unwrap();
        let so = ds
            .filter_pipeline()
            .into_iter()
            .find(|f| f.id == 6)
            .expect("a scale-offset filter");
        // cd_values[7] is FILAVAIL (1 = defined); cd_values[8..] carry the fill
        // value, least-significant 4 bytes per entry.
        (so.client_data[7], so.client_data[8])
    };

    assert_eq!(filavail_and_filval(Some(7), "explicit.h5"), (1, 7));
    assert_eq!(
        filavail_and_filval(None, "default.h5"),
        (1, 0),
        "the default fill value is still a defined one"
    );
}

/// The `cd_values` of `col`, as the file records them.
fn filter_parms(path: &std::path::Path) -> Vec<u32> {
    hdf5_pure::File::open(path)
        .unwrap()
        .dataset("col")
        .unwrap()
        .filter_pipeline()
        .into_iter()
        .find(|f| f.id == 6)
        .expect("a scale-offset filter")
        .client_data
}

/// The other half of the premise above (issue #297): this crate writes the same
/// parameters the reference does, for the same dataset. Every entry is compared,
/// not only the two the fill value occupies — the array is what a decoder reads
/// the chunk through, so one wrong entry anywhere is a chunk neither library
/// decodes as written.
///
/// The widths matter: a `u64` fill value spans **two** `cd_values` entries and
/// every narrower one a single entry with the rest zero, so a writer that
/// packed one entry always, or eight bytes always, agrees with the reference on
/// only part of this.
#[test]
fn a_dataset_this_crate_writes_carries_the_c_librarys_filter_parameters() {
    let dir = tempdir().unwrap();

    macro_rules! compare {
        ($name:literal, $ty:ty, $with:ident, $data:expr, $fill:expr) => {
            compare!(
                $name,
                $ty,
                $with,
                $data,
                $fill,
                CScaleOffset::Integer(0),
                hdf5_pure::ScaleOffset::Integer(0)
            )
        };
        ($name:literal, $ty:ty, $with:ident, $data:expr, $fill:expr, $c_mode:expr, $mode:expr) => {{
            let data: Vec<$ty> = $data;
            let fill: Option<$ty> = $fill;
            let c_path = dir.path().join(concat!($name, "-c.h5"));
            let pure_path = dir.path().join(concat!($name, "-pure.h5"));
            c_create_typed(&c_path, &data, data.len(), $c_mode, fill);
            let mut b = hdf5_pure::FileBuilder::new();
            let ds = b
                .create_dataset("col")
                .$with(&data)
                .with_shape(&[data.len() as u64])
                .with_chunks(&[data.len() as u64])
                .with_scale_offset($mode);
            if let Some(f) = fill {
                ds.with_fill_value(f);
            }
            b.write(&pure_path).unwrap();

            assert_eq!(
                filter_parms(&pure_path),
                filter_parms(&c_path),
                concat!($name, ": cd_values diverge from the C library")
            );
            // The parameters agreeing is not the chunk agreeing: the fill value
            // they record is what the encoder diverts to the sentinel.
            assert_eq!(
                raw_chunks(&pure_path),
                raw_chunks(&c_path),
                concat!($name, ": compressed chunk bytes diverge")
            );
        }};
    }

    // A fill value the caller chose, at each width and signedness. Every fixture
    // holds copies of its own fill value, so the sentinel is exercised too.
    compare!("u8", u8, with_u8_data, vec![7, 1, 2, 7, 3], Some(7));
    compare!("i8", i8, with_i8_data, vec![-9, 1, -2, -9, 3], Some(-9));
    compare!(
        "u16",
        u16,
        with_u16_data,
        vec![700, 1, 2, 700, 3],
        Some(700)
    );
    compare!(
        "i16",
        i16,
        with_i16_data,
        vec![-700, 1, -2, -700, 3],
        Some(-700)
    );
    compare!(
        "u32",
        u32,
        with_u32_data,
        vec![70_000, 1, 2, 70_000, 3],
        Some(70_000)
    );
    compare!(
        "i32",
        i32,
        with_i32_data,
        vec![-70_000, 1, -2, -70_000, 3],
        Some(-70_000)
    );
    // Eight bytes: the fill value spans two `cd_values` entries.
    compare!(
        "u64",
        u64,
        with_u64_data,
        vec![1 << 40, 1, 2, 1 << 40, 3],
        Some(1 << 40)
    );
    compare!(
        "i64",
        i64,
        with_i64_data,
        vec![-(1i64 << 40), 1, -2, -(1i64 << 40), 3],
        Some(-(1i64 << 40))
    );

    // And with no fill value at all, which the reference still records as a
    // defined one, of zero. The zeros in the data are what that turns into a
    // sentinel, so this fixture would pass with the filter parameters right and
    // the encoding wrong if it held none.
    compare!("default", i32, with_i32_data, vec![0, 5, 0, 7, 0], None);

    // Float D-scale is the other emit path through the same parameters, and it
    // matches an element against the fill value within one decimal quantum
    // rather than by equality — so `0.0004` is a fill value for `D = 3` and
    // `0.002` is not. A dataset with no fill value gets the defined zero here
    // too, which is what makes that window apply at all.
    compare!(
        "f64",
        f64,
        with_f64_data,
        vec![-999.0, 1.5, 2.25, -999.0004, 3.125],
        Some(-999.0),
        CScaleOffset::FloatDScale(3),
        hdf5_pure::ScaleOffset::FloatDScale(3)
    );
    compare!(
        "f64-default",
        f64,
        with_f64_data,
        vec![0.0, 1.5, 0.0004, 2.25, -0.0002],
        None,
        CScaleOffset::FloatDScale(3),
        hdf5_pure::ScaleOffset::FloatDScale(3)
    );
    compare!(
        "f32-default",
        f32,
        with_f32_data,
        vec![0.0, 1.5, 0.0004, 2.25, -0.0002],
        None,
        CScaleOffset::FloatDScale(3),
        hdf5_pure::ScaleOffset::FloatDScale(3)
    );
}

/// A fill value the encoder can collapse pays for itself: a chunk that is mostly
/// fill values packs to the width of what is left, not to the width of a range
/// stretched to reach the fill value.
///
/// That saving is the reason the reference records a fill value at all, and it
/// is what this crate could not produce while its writer recorded none.
#[test]
fn a_recorded_fill_value_shrinks_a_mostly_fill_chunk() {
    let dir = tempdir().unwrap();
    // Values near zero, a fill value far from them: without the fill value the
    // chunk's range spans all of it.
    let data: Vec<i32> = (0..64)
        .map(|i| if i % 4 == 0 { i % 3 } else { -1_000_000 })
        .collect();

    let sizes = ["with", "without"].map(|which| {
        let path = dir.path().join(format!("{which}.h5"));
        let mut b = hdf5_pure::FileBuilder::new();
        let ds = b
            .create_dataset("col")
            .with_i32_data(&data)
            .with_shape(&[data.len() as u64])
            .with_chunks(&[data.len() as u64])
            .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0));
        if which == "with" {
            ds.with_fill_value(-1_000_000_i32);
        }
        b.write(&path).unwrap();
        assert_eq!(
            hdf5_pure::File::open(&path)
                .unwrap()
                .dataset("col")
                .unwrap()
                .read_i32()
                .unwrap(),
            data,
            "{which}: the values must survive either way"
        );
        raw_chunks(&path)[0].len()
    });

    assert!(
        sizes[0] * 2 < sizes[1],
        "a recorded fill value should collapse this chunk: {} bytes with, {} without",
        sizes[0],
        sizes[1]
    );
}

/// The issue as filed: append to a C-written scale-offset dataset that carries
/// a defined fill value.
#[test]
fn appending_to_a_c_written_fill_defined_dataset() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("append.h5");
    c_create(&path, &[1, 2, 3], 512, 0);

    pure_append(&path, &[4, 5]);

    assert_eq!(read_c(&path), vec![1, 2, 3, 4, 5]);
    assert_eq!(read_pure(&path), vec![1, 2, 3, 4, 5]);
}

/// Appended elements that *are* the fill value must round-trip: the encoder
/// stores them as the all-ones sentinel, and both decoders must map that back
/// to the fill value rather than to `minval + sentinel`.
#[test]
fn appended_fill_valued_elements_round_trip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("append_fill.h5");
    let fill = 7u64;
    c_create(&path, &[10, 11, 12], 8, fill);

    // A chunk mixing real values with the fill value, then a chunk that is
    // nothing but fill values.
    pure_append(&path, &[20, fill, 21, fill, fill]);
    pure_append(&path, &[fill; 8]);

    let mut want = vec![10, 11, 12, 20, fill, 21, fill, fill];
    want.extend([fill; 8]);
    assert_eq!(read_c(&path), want);
    assert_eq!(read_pure(&path), want);
}

/// The sentinel must not collide with a legitimate offset. A chunk whose
/// non-fill values span exactly the range a naive `minbits` would encode is the
/// case that catches a missing widening: with `minbits = ceil_log2(span)` the
/// largest real offset *is* the all-ones sentinel and decodes as the fill
/// value instead of as itself.
#[test]
fn the_widest_value_in_a_chunk_is_not_mistaken_for_the_fill_value() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("sentinel.h5");
    let fill = 0u64;
    c_create(&path, &[100], 8, fill);

    // min = 100, max = 103: span 4, so an unwidened minbits would be 2 and the
    // offset for 103 would be 0b11 — the sentinel.
    pure_append(&path, &[101, 102, 103, fill, 100, 101, 102]);

    let want = vec![100, 101, 102, 103, fill, 100, 101, 102];
    assert_eq!(read_c(&path), want);
    assert_eq!(read_pure(&path), want);
}

/// Signed data: the fill value comparison and the `value - min` offsets are
/// signed, and a negative `minval` rides in the header as a two's-complement
/// bit pattern.
#[test]
fn signed_data_with_a_negative_fill_value() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("signed.h5");
    let fill = -9i32;
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        let ds = file
            .new_dataset::<i32>()
            .scale_offset(CScaleOffset::Integer(0))
            .chunk((8,))
            .shape((Extent::resizable(3),))
            .fill_value(fill)
            .create("col")
            .unwrap();
        ds.write_raw(&[-100i32, 0, 100]).unwrap();
        file.close().unwrap();
    }

    {
        let f = hdf5_pure::File::open_rw(&path).unwrap();
        f.dataset("col")
            .unwrap()
            .append_staged(|b| {
                b.append_i32(&[-5, fill, 5, fill]);
            })
            .unwrap();
        f.commit().unwrap();
    }

    let want = vec![-100i32, 0, 100, -5, fill, 5, fill];
    assert_eq!(
        hdf5::File::open(&path)
            .unwrap()
            .dataset("col")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        want
    );
    assert_eq!(
        hdf5_pure::File::open(&path)
            .unwrap()
            .dataset("col")
            .unwrap()
            .read_i32()
            .unwrap(),
        want
    );
}

/// Read the raw (still-compressed) bytes of every chunk of `col`, in chunk
/// order, straight out of the file.
fn raw_chunks(path: &std::path::Path) -> Vec<Vec<u8>> {
    let bytes = std::fs::read(path).unwrap();
    let f = hdf5_pure::File::open(path).unwrap();
    let ds = f.dataset("col").unwrap();
    let mut chunks = ds.chunks().unwrap();
    chunks.sort_by_key(|c| c.offset.clone());
    chunks
        .iter()
        .map(|c| {
            let start = usize::try_from(c.address).unwrap();
            let len = usize::try_from(c.storage_size).unwrap();
            bytes[start..start + len].to_vec()
        })
        .collect()
}

/// Semantic agreement is what the tests above check; this checks *byte*
/// agreement, which is stronger. Our encoder is free to pick a `minbits` the C
/// decoder would still read correctly — a chunk that decodes right but packs
/// differently would pass every round-trip above and still mean the two
/// encoders had diverged.
///
/// Create a scale-offset dataset of any supported element type with the C
/// library. `fill` chooses only which *value* the filter records — not whether
/// it records one, which the C library always does; see
/// [`the_c_library_records_a_defined_fill_value_even_without_one`].
fn c_create_typed<T: hdf5::H5Type + Copy>(
    path: &std::path::Path,
    data: &[T],
    chunk: usize,
    mode: CScaleOffset,
    fill: Option<T>,
) {
    let file = hdf5::File::with_options()
        .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
        .create(path)
        .unwrap();
    let mut builder = file
        .new_dataset::<T>()
        .scale_offset(mode)
        .chunk((chunk,))
        .shape((Extent::resizable(data.len()),));
    if let Some(f) = fill {
        builder = builder.fill_value(f);
    }
    let ds = builder.create("col").unwrap();
    if !data.is_empty() {
        ds.write_raw(data).unwrap();
    }
    file.close().unwrap();
}

/// Both files end up holding the same elements in one chunk. In the first the C
/// library encoded it. In the second this crate did, appending to a dataset the
/// C library created — onto a C-written partial chunk under the lossless integer
/// mode, and onto an empty dataset under the lossy float ones, which seed no
/// elements at all (see [`seed_len`]). Either way the compressed bytes must match
/// exactly.
///
/// This is the load-bearing check for everything the encoder decides that a
/// round trip cannot see. Both decoders tolerate a `minbits` a code point too
/// wide, and D-scale is lossy enough to absorb a fill value matched by equality
/// rather than within a quantum — so a value comparison passes either way and
/// only the bytes disagree.
///
/// It sweeps the chunk's value span rather than fixing one: how many bits an
/// element takes is a step function of the span, the two encoders can differ by
/// only a code point or two, and most spans cannot tell a correct reservation
/// from a missing or a doubled one.
///
/// Every fixture's first element is the value passed as the fill, so the copies
/// woven through the rest of the chunk are the ones the encoder must divert to
/// the sentinel.
/// One fixture, one comparison: build the same chunk with the C library and by
/// appending to a C-created dataset, and require the compressed bytes to match.
/// [`assert_byte_identical_encoding`] is this over a swept fixture; a test whose
/// fixture does not vary calls this directly rather than running the identical
/// case twenty times.
fn assert_one_byte_identical_chunk<T, A>(mode: CScaleOffset, values: &[T], append: A)
where
    T: hdf5::H5Type + Copy + PartialEq + std::fmt::Debug,
    A: Fn(&std::path::Path, &[T]),
{
    let dir = tempdir().unwrap();
    let fill = values[0];
    let c_written = dir.path().join("c.h5");
    let appended = dir.path().join("appended.h5");
    c_create_typed(&c_written, values, values.len(), mode, Some(fill));
    let seed = seed_len(mode);
    c_create_typed(&appended, &values[..seed], values.len(), mode, Some(fill));
    append(&appended, &values[seed..]);

    assert_eq!(
        raw_chunks(&appended),
        raw_chunks(&c_written),
        "compressed chunk bytes diverge from the C encoder (fill {fill:?})"
    );
}

/// How many elements the C library writes before this crate appends the rest.
/// The lossless integer mode seeds one, so the append re-encodes a partial
/// chunk; the lossy float modes seed none, because a staged append refuses to
/// re-encode a partial trailing chunk under them (issue #407), and grow the
/// whole chunk from an empty dataset instead.
fn seed_len(mode: CScaleOffset) -> usize {
    match mode {
        CScaleOffset::Integer(_) => 1,
        _ => 0,
    }
}

fn assert_byte_identical_encoding<T, A>(
    mode: CScaleOffset,
    values_for: impl Fn(u64) -> Vec<T>,
    append: A,
) where
    T: hdf5::H5Type + Copy + PartialEq + std::fmt::Debug,
    A: Fn(&std::path::Path, &[T]),
{
    for span in 1..=20u64 {
        let values = values_for(span);
        assert_one_byte_identical_chunk(mode, &values, &append);
    }
}

#[test]
fn unsigned_integer_encoding_matches_the_c_library_byte_for_byte() {
    assert_byte_identical_encoding(
        CScaleOffset::Integer(0),
        |span| {
            let mut v = vec![7u64];
            v.extend(100..100 + span);
            v.push(7);
            v
        },
        pure_append,
    );
}

/// Signed data is the case where the fill value must be compared as a *signed*
/// element: recovered from `cd_values` it is a bit pattern, and a negative fill
/// read as unsigned matches nothing, so every fill element would be encoded as
/// an ordinary offset instead of as the sentinel.
#[test]
fn signed_integer_encoding_matches_the_c_library_byte_for_byte() {
    assert_byte_identical_encoding(
        CScaleOffset::Integer(0),
        |span| {
            let mut v = vec![-9i64];
            v.extend(-50..-50 + span as i64);
            v.push(-9);
            v
        },
        |path, values| {
            let f = hdf5_pure::File::open_rw(path).unwrap();
            f.dataset("col")
                .unwrap()
                .append_staged(|b| {
                    b.append_i64(values);
                })
                .unwrap();
            f.commit().unwrap();
        },
    );
}

/// Float D-scale is where the reference matches an element against the fill
/// value by *proximity* — anything within `10^-D` counts — rather than by
/// equality. `-999.0004` sits inside that window for `D = 3` without being
/// equal to it, so an encoder testing equality diverges here and nowhere else.
#[test]
fn float_dscale_encoding_matches_the_c_library_byte_for_byte() {
    assert_byte_identical_encoding(
        CScaleOffset::FloatDScale(3),
        |span| {
            let mut v = vec![-999.0f64];
            v.extend((0..span).map(|i| 1.5 + i as f64 * 0.25));
            // Within one decimal quantum of the fill value, but not equal.
            v.push(-999.0004);
            v
        },
        pure_append_f64,
    );
}

/// The reference rounds each scaled residual with `llround`, which rounds `x` —
/// not the sum `x + 0.5`, a different function. They part company at the largest
/// double below one half, where the exact sum sits halfway between two doubles
/// and rounds *up*: `llround` answers 0 and the sum answers 1.
///
/// `D = 0` puts that value in front of the rounding unscaled, which is what
/// makes the fixture reach the divergence at all. It costs a decoded value, not
/// only the bytes it is stored as — and through `span` one such element can
/// widen every element in the chunk (issue #300).
#[test]
fn a_residual_just_below_one_half_rounds_the_way_the_c_library_rounds() {
    let below_half = 0.499_999_999_999_999_94f64;
    assert!(
        below_half < 0.5 && below_half + 0.5 == 1.0,
        "the fixture must be a value the two roundings disagree on"
    );
    let values = [-999.0f64, 0.0, below_half, 3.0, -999.0];
    assert_one_byte_identical_chunk(CScaleOffset::FloatDScale(0), &values, pure_append_f64);

    // The `f32` helper is a separate function with the same defect available to
    // it, and the value it parts company with the reference on is its own: the
    // largest float below one half, where the sum rounds to exactly 1.0.
    let below_half_f32 = 0.499_999_97f32;
    assert!(below_half_f32 < 0.5 && below_half_f32 + 0.5 == 1.0);
    assert_one_byte_identical_chunk(
        CScaleOffset::FloatDScale(0),
        &[-999.0f32, 0.0, below_half_f32, 3.0, -999.0],
        pure_append_f32,
    );

    // The same divergence as a value, which is how a user meets it: this crate
    // stored 1.0 for an element the reference stores as 0.0.
    let dir = tempdir().unwrap();
    let path = dir.path().join("value.h5");
    c_create_typed(
        &path,
        &[],
        values.len(),
        CScaleOffset::FloatDScale(0),
        Some(values[0]),
    );
    pure_append_f64(&path, &values);
    assert_eq!(
        hdf5_pure::File::open(&path)
            .unwrap()
            .dataset("col")
            .unwrap()
            .read_f64()
            .unwrap()[2],
        0.0,
        "a residual below one half must round down"
    );
    assert_eq!(
        hdf5::File::open(&path)
            .unwrap()
            .dataset("col")
            .unwrap()
            .read_raw::<f64>()
            .unwrap()[2],
        0.0,
        "and the C library must read back the same element"
    );
}

fn pure_append_f64(path: &std::path::Path, values: &[f64]) {
    let f = hdf5_pure::File::open_rw(path).unwrap();
    f.dataset("col")
        .unwrap()
        .append_staged(|b| {
            b.append_f64(values);
        })
        .unwrap();
    f.commit().unwrap();
}

fn pure_append_f32(path: &std::path::Path, values: &[f32]) {
    let f = hdf5_pure::File::open_rw(path).unwrap();
    f.dataset("col")
        .unwrap()
        .append_staged(|b| {
            b.append_f32(values);
        })
        .unwrap();
    f.commit().unwrap();
}

/// The `f32` half of the D-scale path.
#[test]
fn float32_dscale_encoding_matches_the_c_library_byte_for_byte() {
    assert_byte_identical_encoding(
        CScaleOffset::FloatDScale(3),
        |span| {
            let mut v = vec![-999.0f32];
            v.extend((0..span).map(|i| 1.5 + i as f32 * 0.25));
            v.push(-999.0004);
            v
        },
        pure_append_f32,
    );
}

/// The `f32` path is not the `f64` one at a narrower width. The reference
/// applies its fill-value tolerance at *two different precisions*:
/// `H5Z_scaleoffset_max_min_3` hardcodes `double` `fabs`/`pow` when scanning
/// for the chunk's range, while `H5Z_scaleoffset_modify_1` uses the element
/// type's own `fabsf`/`powf` when rewriting the elements. So there is a window
/// — one `f32` value wide — where an element counts as a fill value for the
/// range scan and as an ordinary value for the encoding.
///
/// `10^-2` is one of the scale factors where that window is non-empty: it is
/// not representable in `f32`, and the `f32` nearest it falls below. An element
/// exactly `f32(0.01)` away from the fill value therefore sits *inside* the
/// `double` window and *outside* the `float` one.
///
/// The chunk this produces decodes to nonsense — the element is excluded from
/// the range it is then encoded relative to — but it decodes to the *same*
/// nonsense in both libraries, which is the point. Reproducing the reference
/// includes reproducing where it is strange.
#[test]
fn the_f32_tolerance_window_between_the_two_precisions_matches() {
    let dir = tempdir().unwrap();
    let fill = 0.0f32;
    // f32(0.01) — below the true 0.01 that the `double` comparison uses, and
    // equal to the `float` one, which the comparison is strict about.
    let edge = 0.01f32;
    assert!(
        f64::from(edge) < 0.01 && edge >= 0.01f32,
        "the fixture must straddle the two tolerances, or it proves nothing"
    );
    let values = vec![fill, 1.5, 1.75, 2.0, 2.25, edge];

    let c_written = dir.path().join("c.h5");
    let appended = dir.path().join("appended.h5");
    let mode = CScaleOffset::FloatDScale(2);
    c_create_typed(&c_written, &values, values.len(), mode, Some(fill));
    c_create_typed(&appended, &[], values.len(), mode, Some(fill));
    pure_append_f32(&appended, &values);

    assert_eq!(raw_chunks(&appended), raw_chunks(&c_written));
}

/// Float D-scale mode with a defined fill value. The reference library matches
/// an element against the fill value within one decimal quantum
/// (`|v - fill| < 10^-D`) rather than by exact equality, so this pins that
/// tolerance as well as the sentinel.
#[test]
fn float_dscale_with_a_defined_fill_value() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("float.h5");
    let decimals = 3u8;
    let fill = -999.0f64;
    {
        let file = hdf5::File::with_options()
            .with_fapl(|p| p.libver_bounds(LibraryVersion::V110, LibraryVersion::latest()))
            .create(&path)
            .unwrap();
        file.new_dataset::<f64>()
            .scale_offset(CScaleOffset::FloatDScale(decimals))
            .chunk((8,))
            .shape((Extent::resizable(0),))
            .fill_value(fill)
            .create("col")
            .unwrap();
        file.close().unwrap();
    }

    // The whole chunk comes from this crate: D-scale is lossy, so a staged
    // append does not re-encode a partial chunk the C library wrote.
    {
        let f = hdf5_pure::File::open_rw(&path).unwrap();
        f.dataset("col")
            .unwrap()
            .append_staged(|b| {
                b.append_f64(&[1.5, 2.25, 3.125, 4.0, fill, 5.5, fill, 6.75]);
            })
            .unwrap();
        f.commit().unwrap();
    }

    let want = [1.5f64, 2.25, 3.125, 4.0, fill, 5.5, fill, 6.75];
    let tol = 0.5 * 10f64.powi(-(i32::from(decimals)));
    for (label, got) in [
        (
            "C",
            hdf5::File::open(&path)
                .unwrap()
                .dataset("col")
                .unwrap()
                .read_raw::<f64>()
                .unwrap(),
        ),
        (
            "pure",
            hdf5_pure::File::open(&path)
                .unwrap()
                .dataset("col")
                .unwrap()
                .read_f64()
                .unwrap(),
        ),
    ] {
        assert_eq!(got.len(), want.len(), "{label} length");
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() <= tol, "{label}: got {g}, want {w}");
        }
    }
}

/// The full-precision fallback: when a chunk's values span so much of the
/// datatype that packing cannot pay, both encoders give up and store the
/// elements verbatim after the header. The header they write is still part of
/// the format, and the reference reaches that path by an early `return` that
/// skips setting `minval` — leaving it at the zero the caller initialized —
/// where the packed paths write the chunk's minimum.
///
/// A decoder ignores `minval` once `minbits` is the full width, so this costs
/// nothing to read either way. It is the kind of difference only a byte
/// comparison sees, which is why the fixture is here rather than in a round
/// trip.
#[test]
fn the_full_precision_fallback_header_matches_the_c_library() {
    // Skipping the fill value leaves min = 1 and max = 255, a span of 254 —
    // past the point where the reference stops packing for a 1-byte type.
    let values: Vec<u8> = vec![0, 200, 1, 255, 0, 200, 1, 255];
    assert_one_byte_identical_chunk(CScaleOffset::Integer(0), &values, |path, v| {
        let f = hdf5_pure::File::open_rw(path).unwrap();
        f.dataset("col")
            .unwrap()
            .append_staged(|b| {
                b.append_raw(v);
            })
            .unwrap();
        f.commit().unwrap();
    });
}

/// The float D-scale counterpart of the fallback above: a chunk whose scaled
/// range overflows the signed integer the offsets are packed into. The
/// reference leaves it at full precision by the same skip-past-`save_min`
/// route, so the header's `minval` is zero rather than the chunk's minimum.
#[test]
fn the_float_full_precision_fallback_header_matches_the_c_library() {
    // The chunk's minimum must not itself be zero, or a header carrying the
    // minimum and one carrying zero are the same bytes.
    let f64_values = vec![-999.0f64, 5.0, 1e19, 6.0, -999.0, 7.0];
    assert_one_byte_identical_chunk(CScaleOffset::FloatDScale(0), &f64_values, |path, v| {
        let f = hdf5_pure::File::open_rw(path).unwrap();
        f.dataset("col")
            .unwrap()
            .append_staged(|b| {
                b.append_f64(v);
            })
            .unwrap();
        f.commit().unwrap();
    });

    let f32_values = vec![-999.0f32, 5.0, 1e10, 6.0, -999.0, 7.0];
    assert_one_byte_identical_chunk(CScaleOffset::FloatDScale(0), &f32_values, pure_append_f32);
}

/// A **fill-undefined** fixture, which the C library's own API cannot create:
/// `H5P_fill_value_defined` answers `DEFAULT` for an untouched fill value, so
/// every dataset it builds records `FILL_DEFINED`. The only route is to let
/// this crate create the dataset — its builder always records
/// `FILL_UNDEFINED` — and then have the C library rewrite the data through the
/// file's own `cd_values`, which re-runs its encoder on them.
///
/// Without this, the whole fill-undefined half of the encoder has no
/// byte-level oracle at all.
fn c_reencode_in_place<T: hdf5::H5Type + Copy>(path: &std::path::Path, data: &[T]) {
    let f = hdf5::File::open_rw(path).unwrap();
    f.dataset("col").unwrap().write_raw(data).unwrap();
    f.close().unwrap();
}

fn first_chunk_header(path: &std::path::Path) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let f = hdf5_pure::File::open(path).unwrap();
    let ds = f.dataset("col").unwrap();
    let c = &ds.chunks().unwrap()[0];
    let s = usize::try_from(c.address).unwrap();
    bytes[s..s + 21].to_vec()
}

/// The full-precision fallback header, at every integer width this crate
/// writes and in both signednesses, against the C library re-encoding the same
/// chunk through the same `cd_values`.
///
/// The reference reaches that path by an early `return` that skips the
/// `*minval = min` its packed paths end with, so every one of these carries a
/// zero where the chunk's minimum would otherwise be — and every fixture's
/// minimum is deliberately non-zero, so an encoder that wrote the minimum here
/// fails rather than agreeing by luck.
///
/// One width is the exception, and only in the *other* fill mode: `signed char`
/// is the one type `H5Z__scaleoffset_precompress_i` hand-expands rather than
/// routing through the `H5Z_scaleoffset_check_2` macro, and the expansion
/// assigns `*minval` in its fill-**undefined** early return. Nothing this crate
/// writes records an undefined fill value any more — the reference records a
/// defined one for every scale-offset dataset, and so does this crate — so that
/// branch is reachable only by re-encoding a file that already carries one, and
/// `the_fill_undefined_fallback_carries_the_signed_char_minimum` covers it
/// against the encoder directly.
///
/// `i32`/`i64` are left out on purpose. The reference's `max - min` overflows
/// signed arithmetic at their extremes, and this build traps inside
/// `H5Dwrite` rather than returning a comparison.
#[test]
fn the_fallback_header_matches_the_c_library_at_every_width() {
    let dir = tempdir().unwrap();

    // Each fixture spreads past `width_max - 2` for its width, so the
    // full-precision fallback fires, with a minimum that is not zero.
    let i8_data: Vec<i8> = vec![-128, 126, -128, 126, -1, -2, -3, -4];
    let u8_data: Vec<u8> = vec![1, 255, 1, 255, 2, 3, 4, 5];
    let i16_data: Vec<i16> = vec![-32768, 32766, -32768, 32766, -1, -2, -3, -4];
    let u16_data: Vec<u16> = vec![1, 65535, 1, 65535, 2, 3, 4, 5];

    let check = |name: &str, want_minval: u64, build: &dyn Fn(&std::path::Path)| {
        let path = dir.path().join(format!("{name}.h5"));
        build(&path);
        let ours = first_chunk_header(&path);
        assert_eq!(
            u32::from_le_bytes(ours[..4].try_into().unwrap()),
            u32::from(name[1..].parse::<u8>().unwrap()),
            "{name}: the fixture must reach the full-precision fallback",
        );
        assert_eq!(
            u64::from_le_bytes(ours[5..13].try_into().unwrap()),
            want_minval,
            "{name}: minval on the fallback path",
        );
        ours
    };

    macro_rules! case {
        ($name:literal, $data:expr, $with:ident, $want:expr) => {{
            let header = check($name, $want, &|p: &std::path::Path| {
                let mut b = hdf5_pure::FileBuilder::new();
                b.create_dataset("col")
                    .$with(&$data)
                    .with_shape(&[$data.len() as u64])
                    .with_chunks(&[$data.len() as u64])
                    .with_scale_offset(hdf5_pure::ScaleOffset::Integer(0));
                b.write(p).unwrap();
            });
            let path = dir.path().join(format!("{}.h5", $name));
            c_reencode_in_place(&path, &$data);
            assert_eq!(
                header,
                first_chunk_header(&path),
                "{}, fill undefined",
                $name
            );
        }};
    }

    // The fill value these are written with is the defined default, so all four
    // take the macro-driven path and carry zero.
    case!("i8", i8_data, with_i8_data, 0);
    case!("u8", u8_data, with_u8_data, 0);
    case!("i16", i16_data, with_i16_data, 0);
    case!("u16", u16_data, with_u16_data, 0);

    // The same holds for a fill value the caller chose, on the one width whose
    // hand-expanded branch could have differed. Non-fill values spread
    // -128..127, past the fallback threshold.
    assert_one_byte_identical_chunk(
        CScaleOffset::Integer(0),
        &[5i8, -128, 127, 5, -128, 127, 0, 1],
        |path, values| {
            let f = hdf5_pure::File::open_rw(path).unwrap();
            f.dataset("col")
                .unwrap()
                .append_staged(|b| {
                    b.append_i8(values);
                })
                .unwrap();
            f.commit().unwrap();
        },
    );
}

/// A chunk that is nothing but fill values has no range to take a minimum
/// over, and the reference leaves both bounds at the zero its locals were
/// declared with rather than at any element's value. Every code in such a
/// chunk is the sentinel, so `minval` is written and never read — which is why
/// a round trip cannot see this and the bytes can.
#[test]
fn an_all_fill_chunk_matches_the_c_library_byte_for_byte() {
    assert_one_byte_identical_chunk(CScaleOffset::Integer(0), &[3u64; 6], pure_append);
    assert_one_byte_identical_chunk(
        CScaleOffset::FloatDScale(3),
        &[2.5f64; 6],
        |path, values| {
            let f = hdf5_pure::File::open_rw(path).unwrap();
            f.dataset("col")
                .unwrap()
                .append_staged(|b| {
                    b.append_f64(values);
                })
                .unwrap();
            f.commit().unwrap();
        },
    );
}

/// `ScaleOffset::Integer(n)` with `n` equal to the datatype's bit width selects
/// the reference's pass-through mode, which it takes *before* splitting on
/// compress versus decompress — so the chunk is stored unfiltered in both
/// directions and carries no header.
///
/// Implementing that on only one side is silent corruption rather than a size
/// difference: this crate read the mode and did not write it, so a chunk it
/// packed came back as raw bytes. The dataset below was unreadable here
/// ("data size mismatch: expected 256 bytes, got 54") and read as garbage by
/// the C library.
#[test]
fn a_full_width_minbits_stores_the_chunk_unfiltered() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("noop.h5");
    let data: Vec<i32> = (0..64).map(|i| 100 + i % 9).collect();
    let mut b = hdf5_pure::FileBuilder::new();
    b.create_dataset("col")
        .with_i32_data(&data)
        .with_shape(&[64])
        .with_chunks(&[64])
        .with_scale_offset(hdf5_pure::ScaleOffset::Integer(32));
    b.write(&path).unwrap();

    // Stored verbatim: one chunk, exactly the element bytes, no 21-byte header.
    let raw = &raw_chunks(&path)[0];
    let expected: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(raw, &expected, "the chunk must be stored unfiltered");

    assert_eq!(
        hdf5_pure::File::open(&path)
            .unwrap()
            .dataset("col")
            .unwrap()
            .read_i32()
            .unwrap(),
        data
    );
    assert_eq!(
        hdf5::File::open(&path)
            .unwrap()
            .dataset("col")
            .unwrap()
            .read_raw::<i32>()
            .unwrap(),
        data
    );
}
