use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::thread;

use ::test_util;
use rstest::rstest;
use test_util::data_layout;
use test_util::object_header::MessageType;
use test_util::object_header::v2;

use super::*;
use crate::DatatypeByteOrder;
use crate::address::StoredAddress;
use crate::address::StoredAddressExt;
use crate::data_layout::{ChunkIndexLayout, ChunkedLayoutFlags};
use crate::filter_pipeline::FilterDescription;
use crate::object_graph::TraversalLimit;
use crate::scaleoffset::{FillAvailability, ScaleOffsetFill};
use crate::source::BytesSource;
use crate::superblock::LocatedSuperblock;

// Private attribute messages expose encoding details omitted by decoded values.
mod attribute_fidelity;

#[rstest]
#[case::reuse(true)]
#[case::read_again(false)]
fn planning_consumes_discovery_headers(#[case] reuse: bool) {
    let mut builder = FileBuilder::new();
    builder.set_attr("root", AttrValue::I32(5));
    let mut group = GroupBuilder::new("g");
    group.set_attr("group", AttrValue::I32(7));
    group
        .create_dataset("a")
        .with_unallocated_storage(crate::make_i32_type(), &[4])
        .set_attr("dataset", AttrValue::I32(11));
    builder.add_group(group.finish());
    let reads = Arc::new(AtomicUsize::new(0));
    let file = Arc::new(
        File::from_source(CountedSource {
            bytes: BytesSource::new(builder.finish().unwrap()),
            reads: Arc::clone(&reads),
        })
        .unwrap(),
    );
    let objects = SourceObjects::discover(&file, &BTreeSet::new(), REPACK_BUDGET).unwrap();
    if !reuse {
        objects.release_headers();
    }
    let source = RepackSource { file, objects };
    reads.store(0, Ordering::Relaxed);
    for index in 0..source.len() {
        drop(source.load(index).unwrap());
    }
    assert_eq!(source.len(), 3);
    assert_eq!(reads.load(Ordering::Relaxed), if reuse { 0 } else { 9 });
    reads.store(0, Ordering::Relaxed);
    for index in 0..source.len() {
        drop(source.load(index).unwrap());
    }
    assert_eq!(reads.load(Ordering::Relaxed), 9);
}

struct CountedSource {
    bytes: BytesSource<Vec<u8>>,
    reads: Arc<AtomicUsize>,
}

impl Source for CountedSource {
    fn len(&self) -> u64 {
        self.bytes.len()
    }
    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), FormatError> {
        let Self { bytes, reads } = self;
        reads.fetch_add(1, Ordering::Relaxed);
        bytes.read_at(offset, out)
    }
}

#[rstest]
#[case::mandatory(false)]
#[case::optional(true)]
fn repack_preserves_an_unallocated_unknown_filter(#[case] optional: bool) {
    let pipeline = test_util::filter_pipeline::user_filter(optional);
    let source = File::from_bytes(unallocated_user_filter_source(optional, &[17], &[8])).unwrap();
    assert_eq!(source.dataset("d").unwrap().chunks().unwrap(), []);

    let destination = File::from_bytes(
        stage(&source, &RepackOptions::new(), REPACK_BUDGET)
            .unwrap()
            .finish()
            .unwrap(),
    )
    .unwrap();
    let dataset = destination.dataset("d").unwrap();
    assert_eq!(dataset.chunks().unwrap(), []);
    assert_eq!(dataset.shape().unwrap(), [17]);
    assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![8]));
    assert_eq!(
        dataset.filter_pipeline_message_bytes().unwrap(),
        Some(pipeline)
    );
    assert_eq!(dataset.read_f64().unwrap(), vec![0.0; 17]);
}

#[rstest]
#[case::mandatory(false)]
#[case::optional(true)]
fn repack_rejects_an_unallocated_contiguous_filter(#[case] optional: bool) {
    let mut bytes = unallocated_user_filter_source(optional, &[17, 1], &[8, 1]);
    let source = File::from_bytes(bytes.clone()).unwrap();
    let header = source.dataset("d").unwrap().header_address().unwrap();
    drop(source);
    v2::replace_message(
        &mut bytes,
        usize::try_from(header).unwrap(),
        MessageType::DATA_LAYOUT,
        &data_layout::contiguous(u64::MAX, 136),
    );
    let source = File::from_bytes(bytes.clone()).unwrap();
    assert_eq!(
        source.dataset("d").unwrap().layout().unwrap(),
        crate::Layout::Contiguous {
            address: None,
            size: 136
        },
    );
    assert_eq!(
        source
            .dataset("d")
            .unwrap()
            .filter_pipeline_message_bytes()
            .unwrap(),
        Some(test_util::filter_pipeline::user_filter(optional)),
    );
    drop(source);
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src.h5");
    let dst = dir.path().join("dst.h5");
    std::fs::write(&src, bytes).unwrap();

    let err = repack(&src, &dst, &RepackOptions::new()).unwrap_err();

    let Error::RepackUnsupported(reason) = &err else {
        panic!("expected RepackUnsupported, got {err:?}");
    };
    assert_eq!(
        reason,
        "dataset d: filter pipeline requires chunked storage"
    );
    assert!(!dst.exists());
}

fn unallocated_user_filter_source(optional: bool, shape: &[u64], chunks: &[u64]) -> Vec<u8> {
    let mut builder = FileBuilder::new();
    let dataset = builder.create_dataset("d");
    dataset
        .with_unallocated_storage(crate::make_f64_type(), shape)
        .with_chunks(chunks);
    dataset.chunk_options.unallocated_pipeline =
        Some(test_util::filter_pipeline::user_filter(optional));
    builder.finish().unwrap()
}

#[rstest]
#[case::objects(TraversalBudget::new(3, 3), TraversalLimit::Objects, 3)]
#[case::links(TraversalBudget::new(4, 2), TraversalLimit::Links, 2)]
fn a_repack_past_its_budget_reports_the_limit_it_exceeds(
    #[case] budget: TraversalBudget,
    #[case] limit: TraversalLimit,
    #[case] maximum: usize,
) {
    let err = stage(&budget_source(), &RepackOptions::new(), budget)
        .map(|_| ())
        .unwrap_err();
    let Error::TraversalLimitExceeded {
        limit: exceeded,
        maximum: allowed,
    } = err
    else {
        panic!("expected TraversalLimitExceeded, got {err:?}");
    };
    assert_eq!((exceeded, allowed), (limit, maximum));
}

#[test]
fn a_repack_within_its_budget_stages_every_object() {
    let bytes = stage(
        &budget_source(),
        &RepackOptions::new(),
        TraversalBudget::new(4, 3),
    )
    .unwrap()
    .finish()
    .unwrap();
    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(file.group("g").unwrap().datasets().unwrap(), ["a", "b"]);
    assert_eq!(file.dataset("g/a").unwrap().read_i32().unwrap(), [1]);
    assert_eq!(file.dataset("g/b").unwrap().read_i32().unwrap(), [2]);
}

#[test]
fn a_repack_rejects_a_root_that_is_not_a_group() {
    let mut builder = FileBuilder::new();
    builder.create_dataset("plain").with_i32_data(&[1]);
    let mut bytes = builder.finish().unwrap();
    let dataset = File::from_bytes(bytes.clone())
        .unwrap()
        .dataset("plain")
        .unwrap()
        .header_address()
        .unwrap();
    let LocatedSuperblock {
        address,
        mut superblock,
        ..
    } = crate::superblock::read(bytes.as_slice()).unwrap();
    assert!(superblock.base_address.is_zero());
    superblock.root_group_address = StoredAddress::new(dataset);
    let rewritten = hdf5_pure_format::__private::serialize_superblock(&superblock).unwrap();
    let at = usize::try_from(address.get()).unwrap();
    bytes[at..at + rewritten.len()].copy_from_slice(&rewritten);

    let err = stage(
        &File::from_bytes(bytes).unwrap(),
        &RepackOptions::new(),
        REPACK_BUDGET,
    )
    .map(|_| ())
    .unwrap_err();

    assert!(
        matches!(&err, Error::NotAGroup(path) if path.is_empty()),
        "{err:?}"
    );
}

#[rstest]
#[case::groups(GroupBuilder::new("g"), vec![])]
#[case::groups_and_a_leaf_dataset(leaf_group(), vec!["leaf"])]
fn a_chain_at_the_depth_bound_repacks_on_a_default_thread_stack(
    #[case] innermost: GroupBuilder,
    #[case] datasets: Vec<&str>,
) {
    let source = chain_source_with(objects::MAX_DEPTH, innermost);
    let bytes = thread::Builder::new()
        .stack_size(DEFAULT_THREAD_STACK)
        .spawn(move || {
            let mut bytes = Vec::new();
            repack_into(&source, &mut bytes, &RepackOptions::new()).unwrap();
            bytes
        })
        .unwrap()
        .join()
        .unwrap();

    let file = File::from_bytes(bytes).unwrap();
    let deepest = file.group(&chain_path(objects::MAX_DEPTH)).unwrap();
    assert_eq!(deepest.groups().unwrap(), Vec::<String>::new());
    assert_eq!(deepest.datasets().unwrap(), datasets);
}

#[rstest]
#[case::groups(GroupBuilder::new("g"))]
#[case::groups_and_a_leaf_dataset(leaf_group())]
fn a_repack_rejects_a_chain_past_the_depth_bound_and_leaves_the_writer_unchanged(
    #[case] innermost: GroupBuilder,
) {
    let mut bytes = WRITER_CONTENT.to_vec();
    let err = repack_into(
        &chain_source_with(objects::MAX_DEPTH + 1, innermost),
        &mut bytes,
        &RepackOptions::new(),
    )
    .unwrap_err();

    let Error::TraversalLimitExceeded { limit, maximum } = err else {
        panic!("expected TraversalLimitExceeded, got {err:?}");
    };
    assert_eq!(
        (limit, maximum),
        (TraversalLimit::Depth, objects::MAX_DEPTH)
    );
    assert_eq!(bytes, WRITER_CONTENT);
}

#[test]
fn a_drop_path_that_shortens_a_chain_brings_it_within_the_depth_bound() {
    let mut bytes = Vec::new();
    repack_into(
        &chain_source(objects::MAX_DEPTH + 1),
        &mut bytes,
        &RepackOptions::new().drop_path(&chain_path(objects::MAX_DEPTH + 1)),
    )
    .unwrap();

    let file = File::from_bytes(bytes).unwrap();
    assert_eq!(
        file.group(&chain_path(objects::MAX_DEPTH))
            .unwrap()
            .groups()
            .unwrap(),
        Vec::<String>::new()
    );
}

#[rstest]
#[case::the_chain("g")]
#[case::inside_the_chain("g/g")]
#[case::the_target("g/g/g/leaf")]
fn a_reference_into_a_dropped_chain_reports_the_path_of_its_target(#[case] dropped: &str) {
    let chain = (1..3).fold(leaf_group().finish(), |inner, _| {
        let mut group = GroupBuilder::new("g");
        group.add_group(inner);
        group.finish()
    });
    let mut builder = FileBuilder::new();
    builder.add_group(chain);
    builder
        .create_dataset("refs")
        .with_path_references(&["g/g/g/leaf"]);
    let source = File::from_bytes(builder.finish().unwrap()).unwrap();

    let err = repack_into(
        &source,
        Vec::new(),
        &RepackOptions::new().drop_path(dropped),
    )
    .unwrap_err();

    let Error::RepackUnsupported(reason) = err else {
        panic!("expected RepackUnsupported, got {err:?}");
    };
    assert_eq!(
        reason,
        "dataset refs: object reference to dropped object \"g/g/g/leaf\" cannot be repacked"
    );
}

fn chain_source(depth: usize) -> File {
    chain_source_with(depth, GroupBuilder::new("g"))
}

fn chain_path(depth: usize) -> String {
    vec!["g"; depth].join("/")
}

fn chain_source_with(depth: usize, innermost: GroupBuilder) -> File {
    let chain = (1..depth).fold(innermost.finish(), |inner, _| {
        let mut group = GroupBuilder::new("g");
        group.add_group(inner);
        group.finish()
    });
    let mut builder = FileBuilder::new();
    builder.add_group(chain);
    File::from_bytes(builder.finish().unwrap()).unwrap()
}

fn leaf_group() -> GroupBuilder {
    let mut group = GroupBuilder::new("g");
    group.create_dataset("leaf").with_i32_data(&[1]);
    group
}

// The root group, `g`, and the two datasets of `g`: four objects and three hard links.
fn budget_source() -> File {
    let mut builder = FileBuilder::new();
    let mut group = builder.create_group("g");
    group.create_dataset("a").with_i32_data(&[1]);
    group.create_dataset("b").with_i32_data(&[2]);
    builder.add_group(group.finish());
    File::from_bytes(builder.finish().unwrap()).unwrap()
}

#[rstest]
#[case(None, DeflateLevel::Six)]
#[case(Some(0), DeflateLevel::Zero)]
#[case(Some(6), DeflateLevel::Six)]
#[case(Some(9), DeflateLevel::Nine)]
fn reconstructs_typed_deflate_level(#[case] raw: Option<u32>, #[case] level: DeflateLevel) {
    let pipeline = deflate_pipeline(raw);
    let filters = check_pipeline(Some(&pipeline), &ObjectPathBuf::parse("data")).unwrap();
    assert_eq!(filters[0].kind, FilterKind::Deflate(level));
}

#[rstest]
#[case(10)]
#[case(255)]
#[case(256)]
#[case(262)]
#[case(u32::MAX)]
fn rejects_invalid_deflate_metadata_without_truncation(#[case] level: u32) {
    let pipeline = deflate_pipeline(Some(level));
    let error = check_pipeline(Some(&pipeline), &ObjectPathBuf::parse("data")).unwrap_err();
    let Error::Format(error) = error else {
        panic!("expected a format error, got {error:?}");
    };
    assert_eq!(error, FormatError::InvalidDeflateLevel { level });
}

fn deflate_pipeline(level: Option<u32>) -> FilterPipeline {
    FilterPipeline {
        version: FILTER_PIPELINE_VERSION_TWO,
        filters: vec![FilterDescription {
            filter_id: FILTER_DEFLATE,
            name: None,
            flags: 0,
            client_data: level.into_iter().collect(),
        }],
    }
}

// Synthetic filter parameters cover both defined and undefined fill availability.
#[test]
fn a_repacked_scale_offset_filter_keeps_the_sources_fill_availability() {
    let ty = crate::scaleoffset::scale_offset_type_from_datatype(&crate::make_i32_type())
        .expect("i32 is a scale-offset type");
    let carried = |fill: ScaleOffsetFill<'_>| {
        let pipeline = Some(FilterPipeline {
            version: 2,
            filters: vec![FilterDescription {
                filter_id: FILTER_SCALEOFFSET,
                name: None,
                flags: 0,
                client_data: scaleoffset::build_cd_values(ScaleOffset::Integer(0), ty, 4, 16, fill)
                    .unwrap(),
            }],
        });
        let mut db = DatasetBuilder::new("d");
        carry_shape_and_pipeline(
            &mut db,
            &[16],
            None,
            &DataLayout::Chunked {
                flags: ChunkedLayoutFlags::NONE,
                // A v3 layout appends the element size, which is what the
                // caller trims back off.
                chunk_dimensions: vec![16, 4],
                index: ChunkIndexLayout::BTreeV1 {
                    address: Some(StoredAddress::new(0x1000)),
                },
            },
            // Routed through `check_pipeline`, as every production caller
            // is: it is the step that decodes the availability out of the
            // source's parameters.
            &check_pipeline(pipeline.as_ref(), &ObjectPathBuf::parse("d"))
                .expect("integer scale-offset repacks"),
        );
        let [
            FilterSpec {
                kind: FilterKind::ScaleOffset(_, fill),
                ..
            },
        ] = db.chunk_options.filters[..]
        else {
            panic!(
                "expected one scale-offset filter, got {:?}",
                db.chunk_options.filters
            );
        };
        fill
    };

    assert_eq!(
        carried(ScaleOffsetFill::Undefined),
        FillAvailability::Undefined
    );
    assert_eq!(
        carried(ScaleOffsetFill::Defined(None)),
        FillAvailability::Defined
    );
}

// A synthetic pipeline represents a combination rejected by the public builder.
#[test]
fn lzf_plus_deflate_pipeline_is_refused() {
    let pipeline = FilterPipeline {
        version: 2,
        filters: vec![
            FilterDescription {
                filter_id: FILTER_LZF,
                name: Some("lzf".into()),
                flags: 0,
                client_data: vec![],
            },
            FilterDescription {
                filter_id: FILTER_DEFLATE,
                name: None,
                flags: 0,
                client_data: vec![6],
            },
        ],
    };
    let err = check_pipeline(Some(&pipeline), &ObjectPathBuf::parse("d")).unwrap_err();
    assert!(
        matches!(&err, Error::RepackUnsupported(msg) if msg.contains("lzf + deflate")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn repack_preserves_big_endian_time_dataset() {
    // Check the complete big-endian TIME datatype and raw element bytes across repacking.

    let dir = tempfile::tempdir().unwrap();
    let dir = dir.path();
    let src = dir.join("hdf5_pure_repack_time_src.h5");
    let dst = dir.join("hdf5_pure_repack_time_dst.h5");

    let dt = Datatype::Time(TimeType::new(4, DatatypeByteOrder::BigEndian, 32).unwrap());
    let raw: Vec<u8> = vec![
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03,
    ];
    {
        let mut b = FileBuilder::new();
        b.create_dataset("t")
            .with_raw_data(dt.clone(), raw.clone(), 3)
            .with_shape(&[3]);
        b.write(&src).unwrap();
    }

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = File::open(&dst).unwrap();
    let ds = f.dataset("t").unwrap();
    assert_eq!(
        ds.datatype().unwrap(),
        dt,
        "time datatype incl. byte order must survive repack"
    );
    assert_eq!(
        ds.read_raw().unwrap(),
        raw,
        "time element bytes must be preserved"
    );

    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&dst).ok();
}

const FILTER_PIPELINE_VERSION_TWO: u8 = 2;

// The stack `std::thread` gives a spawned thread on every Tier 1 platform.
const DEFAULT_THREAD_STACK: usize = 2 << 20;

const WRITER_CONTENT: &[u8] = b"held by the caller";
