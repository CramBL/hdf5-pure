//! Measures metadata decoding and encoding at supported address and length widths.
//!
//! Descriptor benchmarks measure checked construction with writing and writing a reused value.
//! Symbol-table and File Space Info parsers read unaligned input. The repeated descriptor parser
//! returns a fixed-size output array.
//!
//! Fixture construction and correctness checks run outside timing. Returned values are destroyed in
//! each timed iteration. Separate allocation checks use an allocating control.

use std::hint;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion};
use hdf5_pure_core::__private::FileSpaceInfoFields;
use hdf5_pure_core::__private::StoredAddressExt;
use hdf5_pure_core::{
    DecodedStoredAddress, FileSpaceInfo, FileSpacePageSize, FileSpaceStrategy, StoredAddress,
};
use hdf5_pure_format::__private::{
    self as format, Dataspace, DataspaceType, FormatWidths, SymbolTableMessage,
    VariableLengthDescriptor,
};
use hdf5_pure_format::{LengthWidth, MaxExtent, OffsetWidth};
use test_util::allocation;
use test_util::dataspace::{self, Flags, Kind};
use test_util::file_space_info;
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
    let records: Vec<_> = [OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight]
        .into_iter()
        .flat_map(|offsets| {
            [LengthWidth::Two, LengthWidth::Four, LengthWidth::Eight]
                .into_iter()
                .map(move |lengths| FileSpaceFixture::new(FormatWidths { offsets, lengths }))
        })
        .collect();
    let symbols =
        [OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight].map(SymbolTableFixture::new);
    let sequences = [OffsetWidth::Two, OffsetWidth::Four, OffsetWidth::Eight]
        .map(DescriptorSequenceFixture::new);
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
    for fixture in &records {
        fixture.time(&mut criterion);
    }
    for fixture in &symbols {
        fixture.time(&mut criterion);
    }
    for fixture in &sequences {
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
    for fixture in &records {
        fixture.measure();
    }
    for fixture in &symbols {
        fixture.measure();
    }
    for fixture in &sequences {
        fixture.measure();
    }
    println!(
        "native_sizes descriptor={} symbol_table={} file_space_info={}",
        size_of::<VariableLengthDescriptor>(),
        size_of::<SymbolTableMessage>(),
        size_of::<FileSpaceInfo>(),
    );
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
        group.bench_function(
            BenchmarkId::new("construct_and_write", self.width.get()),
            |b| {
                b.iter(|| {
                    let descriptor = VariableLengthDescriptor::new(
                        hint::black_box(self.width),
                        hint::black_box(self.descriptor.element_count()),
                        hint::black_box(self.descriptor.collection_address()),
                        hint::black_box(self.descriptor.object_index()),
                    );
                    let written = descriptor.and_then(|descriptor| {
                        descriptor.write_into(hint::black_box(output.as_mut_slice()))
                    });
                    hint::black_box((&written, &output));
                });
            },
        );
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
        let (written, measured) = allocation::measure(
            &format!(
                "bounded_descriptor_construct_and_write_{}",
                self.width.get()
            ),
            || {
                VariableLengthDescriptor::new(
                    hint::black_box(self.width),
                    hint::black_box(self.descriptor.element_count()),
                    hint::black_box(self.descriptor.collection_address()),
                    hint::black_box(self.descriptor.object_index()),
                )
                .and_then(|descriptor| descriptor.write_into(hint::black_box(&mut output)))
            },
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

struct FileSpaceFixture {
    widths: FormatWidths,
    input: UnalignedInput,
    raw: Vec<u8>,
    expected: FileSpaceInfo,
}

impl FileSpaceFixture {
    fn new(widths: FormatWidths) -> Self {
        let raw = file_space_info::non_persistent_version_1(
            Widths::new(
                usize::from(widths.offsets.get()),
                usize::from(widths.lengths.get()),
            ),
            file_space_info::PAGE,
            64,
            FileSpacePageSize::DEFAULT.get(),
        );
        let fixture = Self {
            widths,
            input: UnalignedInput::new(&raw),
            raw,
            expected: FileSpaceInfoFields {
                strategy: FileSpaceStrategy::Page,
                persist: false,
                threshold: 64,
                page_size: FileSpacePageSize::DEFAULT,
                page_end_meta_threshold: 0,
                eoa_pre_fsm: Some(DecodedStoredAddress::Undefined),
                manager_addrs: Vec::new(),
            }
            .build(),
        };
        assert_eq!(
            format::parse_file_space_info(widths, fixture.input.bytes(fixture.raw.len())),
            Ok(fixture.expected.clone())
        );
        assert_eq!(
            format::serialize_file_space_info_v1(widths, &fixture.expected),
            Ok(fixture.raw.clone())
        );
        fixture
    }

    fn time(&self, criterion: &mut Criterion) {
        let mut group = criterion.benchmark_group("bounded_byte_reads/file_space_info");
        let input = self.input.bytes(self.raw.len());
        group.bench_function(self.id("parse_unaligned"), |b| {
            b.iter(|| {
                hint::black_box(format::parse_file_space_info(
                    hint::black_box(self.widths),
                    hint::black_box(input),
                ))
            });
        });
        group.bench_function(self.id("serialize"), |b| {
            b.iter(|| {
                hint::black_box(format::serialize_file_space_info_v1(
                    hint::black_box(self.widths),
                    hint::black_box(&self.expected),
                ))
            });
        });
        group.finish();
    }

    fn measure(&self) {
        let (parsed, measured) = allocation::measure(&self.id("file_space_parse"), || {
            format::parse_file_space_info(
                hint::black_box(self.widths),
                hint::black_box(self.input.bytes(self.raw.len())),
            )
        });
        assert_eq!(parsed, Ok(self.expected.clone()));
        assert_eq!(measured.blocks, 0);
        assert_eq!(measured.bytes, 0);
        let (encoded, measured) = allocation::measure(&self.id("file_space_serialize"), || {
            format::serialize_file_space_info_v1(
                hint::black_box(self.widths),
                hint::black_box(&self.expected),
            )
        });
        assert_eq!(encoded, Ok(self.raw.clone()));
        assert_eq!(measured.blocks, 1);
        assert_eq!(measured.bytes, u64::try_from(self.raw.len()).unwrap());
        assert_eq!(measured.live_bytes, measured.bytes);
    }

    fn id(&self, operation: &str) -> String {
        format!(
            "{operation}/offsets_{}/lengths_{}",
            self.widths.offsets.get(),
            self.widths.lengths.get()
        )
    }
}

struct SymbolTableFixture {
    width: OffsetWidth,
    input: UnalignedInput,
    len: usize,
    expected: SymbolTableMessage,
}

impl SymbolTableFixture {
    fn new(width: OffsetWidth) -> Self {
        let len = usize::from(width.get());
        let mut raw = 0x1234u64.to_le_bytes()[..len].to_vec();
        raw.extend_from_slice(&0x5678u64.to_le_bytes()[..len]);
        let fixture = Self {
            width,
            input: UnalignedInput::new(&raw),
            len: raw.len(),
            expected: SymbolTableMessage {
                btree_address: StoredAddress::new(0x1234),
                local_heap_address: StoredAddress::new(0x5678),
            },
        };
        assert_eq!(
            SymbolTableMessage::parse_with_offset_width(fixture.input.bytes(fixture.len), width),
            Ok(fixture.expected.clone())
        );
        fixture
    }

    fn time(&self, criterion: &mut Criterion) {
        let mut group = criterion.benchmark_group("bounded_byte_reads/symbol_table");
        let input = self.input.bytes(self.len);
        group.bench_function(BenchmarkId::new("parse_unaligned", self.width.get()), |b| {
            b.iter(|| {
                hint::black_box(SymbolTableMessage::parse_with_offset_width(
                    hint::black_box(input),
                    hint::black_box(self.width),
                ))
            });
        });
        group.finish();
    }

    fn measure(&self) {
        let (parsed, measured) =
            allocation::measure(&format!("symbol_table_parse_{}", self.width.get()), || {
                SymbolTableMessage::parse_with_offset_width(
                    hint::black_box(self.input.bytes(self.len)),
                    hint::black_box(self.width),
                )
            });
        assert_eq!(parsed, Ok(self.expected.clone()));
        assert_eq!(measured.blocks, 0);
        assert_eq!(measured.bytes, 0);
    }
}

struct DescriptorSequenceFixture {
    width: OffsetWidth,
    input: UnalignedInput,
    len: usize,
    expected: [VariableLengthDescriptor; 4],
}

impl DescriptorSequenceFixture {
    fn new(width: OffsetWidth) -> Self {
        let descriptor = DescriptorFixture::new(width);
        let raw = descriptor.raw.repeat(4);
        let fixture = Self {
            width,
            input: UnalignedInput::new(&raw),
            len: raw.len(),
            expected: [descriptor.descriptor; 4],
        };
        assert_eq!(
            Self::parse(width, fixture.input.bytes(fixture.len)),
            Ok(fixture.expected)
        );
        fixture
    }

    fn parse(
        width: OffsetWidth,
        raw: &[u8],
    ) -> Result<[VariableLengthDescriptor; 4], hdf5_pure_format::FormatError> {
        let (first, consumed) = VariableLengthDescriptor::parse(width, raw)?;
        let mut values = [first; 4];
        let mut following = &raw[consumed..];
        for value in values.iter_mut().skip(1) {
            let (parsed, consumed) = VariableLengthDescriptor::parse(width, following)?;
            *value = parsed;
            following = &following[consumed..];
        }
        Ok(values)
    }

    fn time(&self, criterion: &mut Criterion) {
        let mut group = criterion.benchmark_group("bounded_byte_reads/descriptor");
        let input = self.input.bytes(self.len);
        group.bench_function(
            BenchmarkId::new("parse_four_unaligned", self.width.get()),
            |b| {
                b.iter(|| {
                    hint::black_box(Self::parse(
                        hint::black_box(self.width),
                        hint::black_box(input),
                    ))
                });
            },
        );
        group.finish();
    }

    fn measure(&self) {
        let (parsed, measured) =
            allocation::measure(&format!("descriptor_sequence_{}", self.width.get()), || {
                Self::parse(
                    hint::black_box(self.width),
                    hint::black_box(self.input.bytes(self.len)),
                )
            });
        assert_eq!(parsed, Ok(self.expected));
        assert_eq!(measured.blocks, 0);
        assert_eq!(measured.bytes, 0);
    }
}

#[repr(C, align(8))]
struct UnalignedInput([u8; 80]);

impl UnalignedInput {
    fn new(raw: &[u8]) -> Self {
        let mut input = Self([0xcc; 80]);
        input.0[1..1 + raw.len()].copy_from_slice(raw);
        assert_eq!(input.bytes(raw.len()).as_ptr().align_offset(8), 7);
        input
    }

    fn bytes(&self, len: usize) -> &[u8] {
        &self.0[1..1 + len]
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
