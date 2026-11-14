//! Compares bounded fixed records and alignment-one borrowed wire records.
//!
//! Both representations decode into the same owned semantic values. Fixture construction and
//! correctness checks run outside timed iterations. Parsed values are dropped inside each iteration.

#![allow(
    non_ascii_idents,
    reason = "zerocopy derives generate Unicode helper identifiers"
)]

extern crate alloc;

use core::mem;
use std::hint;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion};
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{DecodedStoredAddress, FileSpacePageSize, FileSpaceStrategy, StoredAddress};
use test_util::allocation;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::bytes::Fields;
use crate::error::FormatError;
use crate::fixed_record::{
    BoundDecodedAddress, BoundLength, BoundPageSize, BoundStoredAddress, DecodedAddressCodec,
    FileSpaceStrategyV1Codec, FixedRecord, FixedWire, Length2, Length4, Length8, LengthCodec,
    NonzeroBool, Offset2, Offset4, Offset8, PageSizeCodec, StoredAddressCodec, U16Be, U16Le, U32Le,
    U64Be,
};
use crate::wire::{ZField, ZLength, ZOffset};

#[allow(
    dead_code,
    unused_imports,
    reason = "the benchmark uses private format machinery in its owning source files"
)]
#[path = "../../crates/hdf5-pure-format/src/address.rs"]
mod address;
#[allow(
    dead_code,
    reason = "the benchmark uses the fixed-record subset of the byte cursor"
)]
#[path = "../../crates/hdf5-pure-format/src/bytes.rs"]
mod bytes;
#[allow(
    dead_code,
    reason = "the byte cursor uses a subset of the checked conversions"
)]
#[path = "../../crates/hdf5-pure-format/src/convert.rs"]
mod convert;
#[allow(
    unused_imports,
    reason = "the benchmark shares the format error re-export"
)]
#[path = "../../crates/hdf5-pure-format/src/error.rs"]
mod error;
#[allow(
    dead_code,
    reason = "the benchmark selects representative private field codecs"
)]
#[path = "../../crates/hdf5-pure-format/src/fixed_record/mod.rs"]
mod fixed_record;
#[allow(dead_code, reason = "the byte cursor shares the format width helpers")]
#[path = "../../crates/hdf5-pure-format/src/width.rs"]
mod width;
mod wire;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

macro_rules! record {
    ($codec:ident $(<$($parameter:ident: $bound:path),+>)?
        => $view:ident, $owned:ident for $fields:ident {
        $($field:ident: $field_codec:ty),+ $(,)?
    }) => {
        fixed_record::fixed_record! {
            #[derive(Clone, Copy, Debug)]
            $codec $(<$($parameter: $bound),+>)? => $owned for $fields {
                $($field: $field_codec),+
            }
        }

        #[derive(Clone, Copy, FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned)]
        #[repr(C)]
        struct $view $(<$($parameter: $bound),+>)? {
            $($field: <$field_codec as ZField>::Wire),+
        }

        impl $(<$($parameter: $bound),+>)? ZField for $codec $(<$($parameter),+>)? {
            type Wire = $view $(<$($parameter),+>)?;

            #[inline]
            fn decode_view(wire: &<Self as ZField>::Wire) -> Result<Self::Value, FormatError> {
                Ok($fields { $($field: <$field_codec as ZField>::decode_view(&wire.$field)?),+ })
            }

            #[inline]
            fn encode_view(wire: &mut <Self as ZField>::Wire, value: &Self::Value) {
                let $fields { $($field),+ } = value;
                $(<$field_codec as ZField>::encode_view(&mut wire.$field, $field);)+
            }
        }
    };
}

record! {
    Descriptor<O: ZOffset> => DescriptorView, DescriptorWire for DescriptorFields {
        element_count: U32Le,
        collection_address: StoredAddressCodec<O>,
        object_index: U32Le,
    }
}

record! {
    Symbol<O: ZOffset> => SymbolView, SymbolWire for SymbolFields {
        btree_address: StoredAddressCodec<O>,
        local_heap_address: StoredAddressCodec<O>,
    }
}

record! {
    Prefix<O: ZOffset, L: ZLength> => PrefixView, PrefixWire for PrefixFields {
        strategy: FileSpaceStrategyV1Codec,
        persist: NonzeroBool,
        threshold: LengthCodec<L>,
        page_size: PageSizeCodec<L>,
        page_end_meta_threshold: U16Le,
        eoa_pre_fsm: DecodedAddressCodec<O>,
    }
}

record! {
    Pair => PairView, PairWire for PairFields {
        little: U32Le,
        big: U64Be,
    }
}

record! {
    Envelope => EnvelopeView, EnvelopeWire for EnvelopeFields {
        prefix: U16Be,
        pair: Pair,
        suffix: U16Le,
    }
}

fn main() {
    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();
    allocation_control();
    offset::<Offset2>(&mut criterion);
    offset::<Offset4>(&mut criterion);
    offset::<Offset8>(&mut criterion);
    nested(&mut criterion);
    criterion.final_summary();
}

fn offset<O: ZOffset>(criterion: &mut Criterion) {
    let mut descriptor = 0x1234_5678u32.to_le_bytes().to_vec();
    descriptor.extend_from_slice(&0x1234u64.to_le_bytes()[..usize::from(O::WIDTH.get())]);
    descriptor.extend_from_slice(&0x90ab_cdefu32.to_le_bytes());
    measure::<Descriptor<O>>(
        criterion,
        &format!("descriptor/{}", O::WIDTH.get()),
        &descriptor,
        || {
            Ok(DescriptorFields {
                element_count: hint::black_box(0x1234_5678),
                collection_address: BoundStoredAddress::new(hint::black_box(StoredAddress::new(
                    0x1234,
                )))?,
                object_index: hint::black_box(0x90ab_cdef),
            })
        },
    );
    let mut symbol = 0x1234u64.to_le_bytes()[..usize::from(O::WIDTH.get())].to_vec();
    symbol.extend_from_slice(&0x5678u64.to_le_bytes()[..usize::from(O::WIDTH.get())]);
    measure::<Symbol<O>>(
        criterion,
        &format!("symbol/{}", O::WIDTH.get()),
        &symbol,
        || {
            Ok(SymbolFields {
                btree_address: BoundStoredAddress::new(hint::black_box(StoredAddress::new(
                    0x1234,
                )))?,
                local_heap_address: BoundStoredAddress::new(hint::black_box(StoredAddress::new(
                    0x5678,
                )))?,
            })
        },
    );
    prefix::<O, Length2>(criterion);
    prefix::<O, Length4>(criterion);
    prefix::<O, Length8>(criterion);
}

fn prefix<O: ZOffset, L: ZLength>(criterion: &mut Criterion) {
    let mut raw = vec![1, 1];
    raw.extend_from_slice(&64u64.to_le_bytes()[..usize::from(L::WIDTH.get())]);
    raw.extend_from_slice(&513u64.to_le_bytes()[..usize::from(L::WIDTH.get())]);
    raw.extend_from_slice(&7u16.to_le_bytes());
    raw.extend_from_slice(&u64::MAX.to_le_bytes()[..usize::from(O::WIDTH.get())]);
    let page_size = FileSpacePageSize::try_from(513).unwrap();
    measure::<Prefix<O, L>>(
        criterion,
        &format!("prefix/offset_{}/length_{}", O::WIDTH.get(), L::WIDTH.get()),
        &raw,
        || {
            Ok(PrefixFields {
                strategy: hint::black_box(FileSpaceStrategy::Page),
                persist: hint::black_box(true),
                threshold: BoundLength::new(hint::black_box(64))?,
                page_size: BoundPageSize::new(hint::black_box(page_size))?,
                page_end_meta_threshold: hint::black_box(7),
                eoa_pre_fsm: BoundDecodedAddress::new(hint::black_box(
                    DecodedStoredAddress::Undefined,
                ))?,
            })
        },
    );
}

fn nested(criterion: &mut Criterion) {
    let raw = [
        0x12, 0x34, 0x78, 0x56, 0x34, 0x12, 1, 2, 3, 4, 5, 6, 7, 8, 0xbc, 0x9a,
    ];
    measure::<Envelope>(criterion, "nested", &raw, || {
        Ok(EnvelopeFields {
            prefix: hint::black_box(0x1234),
            pair: PairFields {
                little: hint::black_box(0x1234_5678),
                big: hint::black_box(0x0102_0304_0506_0708),
            },
            suffix: hint::black_box(0x9abc),
        })
    });
}

fn measure<C: ZField>(
    criterion: &mut Criterion,
    name: &str,
    raw: &[u8],
    prepare: impl Fn() -> Result<C::Value, FormatError>,
) where
    C::Value: core::fmt::Debug,
{
    let mut backing = Aligned([0xcc; 128]);
    backing.0[1..raw.len() + 1].copy_from_slice(raw);
    let input = &backing.0[1..raw.len() + 3];
    assert_eq!(input.as_ptr().addr() % 8, 1);
    assert_eq!(mem::align_of::<<C as ZField>::Wire>(), 1);
    assert_eq!(
        mem::size_of::<<C as ZField>::Wire>(),
        <C as FixedRecord>::Wire::SIZE
    );
    let prepared = prepare().unwrap();
    let expected = encoded::<C>(&prepared);
    assert_eq!(expected, raw);
    for reader in [read::<C>, read_view::<C>] {
        let (value, consumed) = reader(input).unwrap();
        assert_eq!(encoded::<C>(&value), expected);
        assert_eq!(consumed, raw.len());
        for end in 0..raw.len() {
            assert_eq!(
                reader(&input[..end]).unwrap_err(),
                FormatError::UnexpectedEof {
                    expected: raw.len(),
                    available: end,
                }
            );
        }
    }
    for writer in [C::write, C::write_view] {
        let mut output = [0xcc; 128];
        assert_eq!(writer(&mut output[1..], &prepared).unwrap(), raw.len());
        assert_eq!(&output[1..raw.len() + 1], expected);
        assert!(output[raw.len() + 1..].iter().all(|&byte| byte == 0xcc));
        for end in 0..raw.len() {
            let mut short = [0xcc; 128];
            assert_eq!(
                writer(&mut short[1..end + 1], &prepared).unwrap_err(),
                FormatError::UnexpectedEof {
                    expected: raw.len(),
                    available: end,
                }
            );
            assert_eq!(short, [0xcc; 128]);
        }
    }
    let mut group = criterion.benchmark_group(format!("fixed_records/{name}"));
    measure_variant::<C, false>(&mut group, input, &prepared, &prepare, &expected, name);
    measure_variant::<C, true>(&mut group, input, &prepared, &prepare, &expected, name);
    group.finish();
    repeated::<C>(criterion, name, raw, &expected);
    println!(
        "fixed_record_size {name} semantic={} wire={} align={}",
        mem::size_of::<C::Value>(),
        mem::size_of::<<C as ZField>::Wire>(),
        mem::align_of::<<C as ZField>::Wire>()
    );
}

fn measure_variant<C: ZField, const VIEW: bool>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    input: &[u8],
    prepared: &C::Value,
    prepare: &impl Fn() -> Result<C::Value, FormatError>,
    expected: &[u8],
    name: &str,
) {
    let label = if VIEW { "zerocopy" } else { "bounded" };
    let reader = |raw: &[u8]| {
        if VIEW {
            read_view::<C>(raw)
        } else {
            read::<C>(raw)
        }
    };
    let writer = |output: &mut [u8], value: &C::Value| {
        if VIEW {
            C::write_view(output, value)
        } else {
            C::write(output, value)
        }
    };
    group.bench_function(BenchmarkId::new("parse_unaligned", label), |b| {
        b.iter(|| hint::black_box(reader(hint::black_box(input))));
    });
    let mut output = [0xcc; 128];
    group.bench_function(BenchmarkId::new("write_prepared", label), |b| {
        b.iter(|| {
            let result = writer(hint::black_box(&mut output[1..]), hint::black_box(prepared));
            hint::black_box((&result, &output));
        });
    });
    group.bench_function(BenchmarkId::new("prepare_and_write", label), |b| {
        b.iter(|| {
            let result =
                prepare().and_then(|value| writer(hint::black_box(&mut output[1..]), &value));
            hint::black_box((&result, &output));
        });
    });
    let (value, measured) = allocation::measure(&format!("{name}_{label}_parse"), || {
        reader(hint::black_box(input))
    });
    assert_eq!(encoded::<C>(&value.unwrap().0), expected);
    assert_eq!((measured.blocks, measured.bytes), (0, 0));
    let (written, measured) = allocation::measure(&format!("{name}_{label}_write"), || {
        prepare().and_then(|value| writer(hint::black_box(&mut output[1..]), &value))
    });
    assert_eq!(written.unwrap(), expected.len());
    assert_eq!((measured.blocks, measured.bytes), (0, 0));
}

#[inline]
fn read<C: FixedRecord>(raw: &[u8]) -> Result<(C::Value, usize), FormatError> {
    let mut fields = Fields::new(raw, 0);
    let value = C::read(&mut fields)?;
    Ok((value, fields.pos()))
}

#[inline]
fn read_view<C: ZField>(raw: &[u8]) -> Result<(C::Value, usize), FormatError> {
    let (wire, _) = <C as ZField>::Wire::ref_from_prefix(raw).map_err(|_layout_error| {
        FormatError::UnexpectedEof {
            expected: <C as FixedRecord>::Wire::SIZE,
            available: raw.len(),
        }
    })?;
    Ok((C::decode_view(wire)?, <C as FixedRecord>::Wire::SIZE))
}

fn encoded<C: FixedRecord>(value: &C::Value) -> Vec<u8> {
    let mut output = vec![0; <C as FixedRecord>::Wire::SIZE];
    assert_eq!(C::write(&mut output, value).unwrap(), output.len());
    output
}

fn repeated<C: ZField>(criterion: &mut Criterion, name: &str, raw: &[u8], expected: &[u8]) {
    let repeated = raw.repeat(4);
    let mut input = Aligned([0xcc; 128]);
    input.0[1..repeated.len() + 1].copy_from_slice(&repeated);
    let raw = &input.0[1..repeated.len() + 3];
    let mut group = criterion.benchmark_group(format!("fixed_records/{name}/repeated_4"));
    repeated_variant::<C, false>(&mut group, raw, expected, name);
    repeated_variant::<C, true>(&mut group, raw, expected, name);
    group.finish();
}

fn repeated_variant<C: ZField, const VIEW: bool>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    raw: &[u8],
    expected: &[u8],
    name: &str,
) {
    let reader = |raw: &[u8], count: usize| {
        if VIEW {
            repeated_views::<C>(raw, count)
        } else {
            repeated_bounded::<C>(raw, count)
        }
    };
    let (values, consumed) = reader(raw, 4).unwrap();
    assert_eq!(values.len(), 4);
    assert_eq!(consumed, expected.len() * 4);
    for value in &values {
        assert_eq!(encoded::<C>(value), expected);
    }
    assert_eq!(
        reader(raw, usize::MAX).err().unwrap(),
        FormatError::UnexpectedEof {
            expected: usize::MAX,
            available: raw.len(),
        }
    );
    assert_eq!(reader(raw, 0).unwrap().1, 0);
    for end in 0..expected.len() * 4 {
        assert_eq!(
            reader(&raw[..end], 4).err().unwrap(),
            FormatError::UnexpectedEof {
                expected: expected.len() * 4,
                available: end,
            }
        );
    }
    let label = if VIEW { "zerocopy" } else { "bounded" };
    group.bench_function(label, |b| {
        b.iter(|| hint::black_box(reader(hint::black_box(raw), hint::black_box(4))))
    });
    let (result, measured) = allocation::measure(&format!("{name}_repeated_{label}"), || {
        reader(hint::black_box(raw), hint::black_box(4))
    });
    assert_eq!(result.unwrap().1, expected.len() * 4);
    assert_eq!(measured.blocks, 1);
    assert!(measured.bytes >= u64::try_from(4 * mem::size_of::<C::Value>()).unwrap());
}

fn repeated_bounded<C: FixedRecord>(
    raw: &[u8],
    count: usize,
) -> Result<(Vec<C::Value>, usize), FormatError> {
    let bytes = repeated_extent::<C>(raw, count)?;
    let mut values = Vec::with_capacity(count);
    for record in bytes.chunks_exact(<C as FixedRecord>::Wire::SIZE) {
        values.push(C::read(&mut Fields::new(record, 0))?);
    }
    Ok((values, bytes.len()))
}

fn repeated_views<C: ZField>(
    raw: &[u8],
    count: usize,
) -> Result<(Vec<C::Value>, usize), FormatError> {
    let bytes = repeated_extent::<C>(raw, count)?;
    let views = <[<C as ZField>::Wire]>::ref_from_bytes(bytes).map_err(|_layout_error| {
        FormatError::Internal("fixed-record extent differs from its wire layout".into())
    })?;
    let mut values = Vec::with_capacity(count);
    for view in views {
        values.push(C::decode_view(view)?);
    }
    Ok((values, bytes.len()))
}

fn repeated_extent<C: FixedRecord>(raw: &[u8], count: usize) -> Result<&[u8], FormatError> {
    let Some(size) = count.checked_mul(<C as FixedRecord>::Wire::SIZE) else {
        return Err(FormatError::UnexpectedEof {
            expected: usize::MAX,
            available: raw.len(),
        });
    };
    let mut fields = Fields::new(raw, 0);
    Ok(fields.take_arrays::<1>(size)?.as_flattened())
}

fn allocation_control() {
    let (control, measured) = allocation::measure("fixed_records_positive_control", || {
        vec![hint::black_box(0x5au8); hint::black_box(256)]
    });
    assert_eq!(control, vec![0x5a; 256]);
    assert!(measured.blocks > 0);
    assert!(measured.bytes >= 256);
}

#[repr(align(8))]
struct Aligned([u8; 128]);
