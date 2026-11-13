//! Measures descriptor parsing and writing, and version 2 dataspace parsing.
//!
//! Fixture construction and output checks run outside timing. Each timed parsing iteration includes
//! result destruction. Separate allocation checks measure the output vectors against an allocating
//! control.

use std::hint;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion};
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::StoredAddress;
use hdf5_pure_format::__private::{Dataspace, DataspaceType, VariableLengthDescriptor};
use hdf5_pure_format::{LengthWidth, MaxExtent, OffsetWidth};
use test_util::allocation;
use test_util::dataspace::{self, Flags, Kind};
use test_util::widths::Widths;

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

fn main() {
    let descriptors =
        [OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight].map(DescriptorFixture::new);
    let dataspaces: Vec<_> = [LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight]
        .into_iter()
        .flat_map(|width| {
            [0, 1, 4, 32].into_iter().flat_map(move |rank| {
                [false, true]
                    .into_iter()
                    .map(move |maxima| DataspaceFixture::new(width, rank, maxima))
            })
        })
        .collect();
    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
        .configure_from_args();
    for fixture in &descriptors {
        fixture.time(&mut criterion);
    }
    for fixture in &dataspaces {
        fixture.time(&mut criterion);
    }
    criterion.final_summary();
    allocation_control();
    for fixture in &descriptors {
        fixture.measure();
    }
    for fixture in &dataspaces {
        fixture.measure();
    }
}

struct DescriptorFixture {
    width: OffsetWidth,
    raw: Vec<u8>,
    descriptor: VariableLengthDescriptor,
}

impl DescriptorFixture {
    fn new(width: OffsetWidth) -> Self {
        let mut raw = 0x1234_5678u32.to_le_bytes().to_vec();
        raw.extend_from_slice(&0x1234u64.to_le_bytes()[..usize::from(width.get())]);
        raw.extend_from_slice(&0x90ab_cdefu32.to_le_bytes());
        // The address 0x1234 fits every supported offset width.
        let descriptor = VariableLengthDescriptor::new(
            width,
            0x1234_5678,
            StoredAddress::new(0x1234),
            0x90ab_cdef,
        )
        .unwrap();
        let fixture = Self {
            width,
            raw,
            descriptor,
        };
        fixture.assert_parse(VariableLengthDescriptor::parse(width, &fixture.raw));
        let mut output = [0xcc; 18];
        fixture.assert_write(descriptor.write_into(&mut output), &output);
        fixture
    }

    fn time(&self, criterion: &mut Criterion) {
        let mut group = criterion.benchmark_group("bounded_byte_reads/descriptor");
        group.bench_function(BenchmarkId::new("parse", self.width.get()), |b| {
            b.iter(|| {
                hint::black_box(VariableLengthDescriptor::parse(
                    hint::black_box(self.width),
                    hint::black_box(self.raw.as_slice()),
                ))
            });
        });
        let mut output = [0xcc; 18];
        group.bench_function(BenchmarkId::new("write_into", self.width.get()), |b| {
            b.iter(|| {
                let written = hint::black_box(self.descriptor)
                    .write_into(hint::black_box(output.as_mut_slice()));
                hint::black_box((&written, &output));
            });
        });
        group.finish();
    }

    fn measure(&self) {
        let (parsed, measured) = allocation::measure(
            &format!("bounded_descriptor_parse_{}", self.width.get()),
            || {
                VariableLengthDescriptor::parse(
                    hint::black_box(self.width),
                    hint::black_box(self.raw.as_slice()),
                )
            },
        );
        self.assert_parse(parsed);
        assert_eq!(measured.blocks, 0);
        assert_eq!(measured.bytes, 0);
        let mut output = [0xcc; 18];
        let (written, measured) = allocation::measure(
            &format!("bounded_descriptor_write_{}", self.width.get()),
            || hint::black_box(self.descriptor).write_into(hint::black_box(&mut output)),
        );
        self.assert_write(written, &output);
        assert_eq!(measured.blocks, 0);
        assert_eq!(measured.bytes, 0);
    }

    fn assert_parse(
        &self,
        parsed: Result<(VariableLengthDescriptor, usize), hdf5_pure_format::FormatError>,
    ) {
        assert_eq!(parsed, Ok((self.descriptor, self.raw.len())));
    }

    fn assert_write(
        &self,
        written: Result<usize, hdf5_pure_format::FormatError>,
        output: &[u8; 18],
    ) {
        assert_eq!(written, Ok(self.raw.len()));
        assert_eq!(&output[..self.raw.len()], self.raw);
        assert!(output[self.raw.len()..].iter().all(|&byte| byte == 0xcc));
    }
}

struct DataspaceFixture {
    width: LengthWidth,
    raw: Vec<u8>,
    expected: Dataspace,
}

impl DataspaceFixture {
    fn new(width: LengthWidth, rank: u8, maxima: bool) -> Self {
        let dimensions: Vec<_> = (0..rank)
            .map(|dimension| u64::from(dimension) + 1)
            .collect();
        let maximum: Vec<_> = dimensions.iter().map(|dimension| dimension + 64).collect();
        let raw = dataspace::v2_with_widths(
            rank,
            if maxima { Flags::MAX_DIMS } else { Flags::NONE },
            if rank == 0 {
                Kind::SCALAR
            } else {
                Kind::SIMPLE
            },
            &dimensions,
            maxima.then_some(maximum.as_slice()),
            Widths::new(8, usize::from(width.get())),
        );
        let fixture = Self {
            width,
            raw,
            expected: Dataspace {
                space_type: if rank == 0 {
                    DataspaceType::Scalar
                } else {
                    DataspaceType::Simple
                },
                rank,
                dimensions,
                max_dimensions: maxima.then(|| maximum.into_iter().map(MaxExtent::Fixed).collect()),
            },
        };
        assert_eq!(
            Dataspace::parse_with_length_width(&fixture.raw, width),
            Ok(fixture.expected.clone()),
        );
        fixture
    }

    fn time(&self, criterion: &mut Criterion) {
        let mut group = criterion.benchmark_group("bounded_byte_reads/dataspace");
        group.bench_function(self.id(), |b| {
            b.iter(|| {
                hint::black_box(Dataspace::parse_with_length_width(
                    hint::black_box(self.raw.as_slice()),
                    hint::black_box(self.width),
                ))
            });
        });
        group.finish();
    }

    fn measure(&self) {
        let (parsed, measured) = allocation::measure(&self.id(), || {
            Dataspace::parse_with_length_width(
                hint::black_box(self.raw.as_slice()),
                hint::black_box(self.width),
            )
        });
        assert_eq!(parsed, Ok(self.expected.clone()));
        let count = u64::from(self.expected.rank != 0)
            * (1 + u64::from(self.expected.max_dimensions.is_some()));
        assert_eq!(measured.blocks, count);
        let bytes = u64::from(self.expected.rank)
            * (size_of::<u64>() as u64
                + if self.expected.max_dimensions.is_some() {
                    size_of::<MaxExtent>() as u64
                } else {
                    0
                });
        assert_eq!(measured.bytes, bytes);
        assert_eq!(measured.peak_bytes, bytes);
        assert_eq!(measured.live_bytes, bytes);
    }

    fn id(&self) -> String {
        format!(
            "parse/width_{}/rank_{}/maxima_{}",
            self.width.get(),
            self.expected.rank,
            self.expected.max_dimensions.is_some(),
        )
    }
}

fn allocation_control() {
    let (control, measured) = allocation::measure("bounded_byte_reads_control", || {
        hint::black_box(vec![hint::black_box(0xa5u8); hint::black_box(1024)])
    });
    assert_eq!(control, vec![0xa5; 1024]);
    assert_eq!(measured.blocks, 1);
    assert_eq!(measured.bytes, 1024);
    assert_eq!(measured.peak_bytes, 1024);
    assert_eq!(measured.live_bytes, 1024);
}
