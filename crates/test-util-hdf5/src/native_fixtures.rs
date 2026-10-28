//! Native file constructors and fixture observations.
//!
//! Constructors close their files before returning. Storage assertions follow
//! addresses read from each file.
//!
//! # Examples
//!
//! ```
//! use hdf5_pure::File;
//! use test_util::temp;
//! use test_util_hdf5::native_fixtures::Fixture;
//! use test_util_hdf5::numeric::ByteOrder;
//!
//! let fixture = Fixture::HardLinkUndo;
//! let path = temp::temp_path(fixture.file_name());
//! fixture.write(&path);
//! let order = if cfg!(target_endian = "big") {
//!     ByteOrder::BigEndian
//! } else {
//!     ByteOrder::LittleEndian
//! };
//! fixture.assert_native(&path, order);
//! let bytes = std::fs::read(&path)?;
//! let file = File::open(&path)?;
//! fixture.assert_contents(&file, order);
//! fixture.assert_representation(&file, &bytes);
//! file.close()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::HashMap;
use std::fmt::Debug;
use std::path::Path;

use hdf5::Dataset as NativeDataset;
use hdf5::Datatype as NativeDatatype;
use hdf5::FileBuilder as NativeFileBuilder;
use hdf5::GroupStorageType;
use hdf5::H5Type;
use hdf5::SimpleExtents;
use hdf5::dataset::Layout as NativeLayout;
use hdf5::file::LibraryVersion;
use hdf5::plist::file_create::FileSpaceStrategy;
use hdf5_pure::AttrValue;
use hdf5_pure::ChunkIndex;
use hdf5_pure::Datatype;
use hdf5_pure::File;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceStrategy as PureStrategy;
use hdf5_pure::FixedPointType;
use hdf5_pure::FloatingPointType;
use hdf5_pure::FreeSpaceManagerRole;
use hdf5_pure::Layout;
use hdf5_pure::MessageType;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::ChunkIndexLayout;
use hdf5_pure_format::__private::DataLayout;
use hdf5_pure_format::__private::ExtensibleArrayHeader;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FractalHeapHeader;
use hdf5_pure_format::__private::LinkInfoMessage;
use hdf5_pure_format::__private::ObjectHeader;

use crate::numeric::ByteOrder as NumericOrder;
use crate::storage;
use crate::symbol_table;

/// A native fixture with a selected storage or object-identity premise.
#[derive(Clone, Copy, Debug)]
pub enum Fixture {
    /// A committed datatype in a version 1 header, shared by two datasets.
    CommittedDatatypeV1,
    /// A continued object header and a fragment in the generic-large free-space manager.
    GenericLarge,
    /// Two hard links to one dataset.
    HardLinkUndo,
    /// Twelve links stored as huge fractal-heap objects.
    HugeLinks12,
    /// An extensible-array chunk index and raw chunks on separate pages.
    PagedIndex,
}

impl Fixture {
    /// Returns the file name shared by generated and committed copies.
    pub fn file_name(self) -> &'static str {
        match self {
            Self::CommittedDatatypeV1 => "committed_datatype_v1.h5",
            Self::GenericLarge => "generic_large.h5",
            Self::HardLinkUndo => "hard_link_undo.h5",
            Self::HugeLinks12 => "huge_links_12.h5",
            Self::PagedIndex => "paged_index.h5",
        }
    }

    /// Writes the selected fixture through the reference library and closes the file.
    ///
    /// # Panics
    ///
    /// Panics if native creation, writing, or close fails.
    pub fn write(self, path: &Path) {
        match self {
            Self::CommittedDatatypeV1 => committed_datatype_v1(path),
            Self::GenericLarge => generic_large(path),
            Self::HardLinkUndo => hard_link_undo(path),
            Self::HugeLinks12 => huge_links_12(path),
            Self::PagedIndex => paged_index(path),
        }
    }

    /// Checks metadata, values, and object relationships through the reference library.
    ///
    /// Numeric datatypes must use `order`.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be opened, metadata or values cannot be read,
    /// any observation differs from the fixture, or close fails.
    pub fn assert_native(self, path: &Path, order: NumericOrder) {
        let file = crate::file::libhdf5_open(path);
        let mut names = file.member_names().unwrap();
        names.sort();
        assert_eq!(names, self.root_names());
        assert!(file.attr_names().unwrap().is_empty());
        match self {
            Self::CommittedDatatypeV1 => {
                let datatype = file.committed_datatype("mytype").unwrap();
                assert_eq!(
                    datatype.as_datatype().to_descriptor().unwrap(),
                    i32::type_descriptor()
                );
                assert_eq!(datatype.as_datatype().byte_order(), order.native_order());
                assert_eq!(datatype.loc_info().unwrap().num_links, 3);
                for name in ["first", "second"] {
                    let dataset = file.dataset(name).unwrap();
                    assert_native_values(&dataset, &[1i32, 2], order);
                    assert!(dataset.dtype().unwrap().is_committed());
                }
            }
            Self::GenericLarge => {
                let dataset = file.dataset("d").unwrap();
                assert_native_values(&dataset, &[1.0f64; 64], order);
                assert_eq!(dataset.attr_names().unwrap(), ["big_attr"]);
                let attribute = dataset.attr("big_attr").unwrap();
                assert_eq!(attribute.shape(), [512]);
                assert_eq!(
                    attribute.dtype().unwrap().to_descriptor().unwrap(),
                    i64::type_descriptor()
                );
                assert_eq!(
                    attribute.dtype().unwrap().byte_order(),
                    order.native_order()
                );
                assert_eq!(attribute.read_raw::<i64>().unwrap(), [7; 512]);
            }
            Self::HardLinkUndo => {
                for name in ["aa", "bb"] {
                    assert_native_values(&file.dataset(name).unwrap(), &[1i32, 2, 3], order);
                }
                let aa = file.loc_info_by_name("aa").unwrap();
                let bb = file.loc_info_by_name("bb").unwrap();
                assert_eq!(aa.token, bb.token);
                assert_eq!(aa.num_links, 2);
                assert_eq!(bb.num_links, 2);
            }
            Self::HugeLinks12 => {
                let group = file.group("g").unwrap();
                assert!(group.attr_names().unwrap().is_empty());
                assert_eq!(group.info().unwrap().storage_type, GroupStorageType::Dense);
                let mut names = group.member_names().unwrap();
                names.sort();
                let mut expected: Vec<_> = (0..12).map(huge_link_name).collect();
                expected.sort();
                assert_eq!(names, expected);
                for index in 0..12 {
                    assert_native_values(
                        &group.dataset(&huge_link_name(index)).unwrap(),
                        &[index],
                        order,
                    );
                }
            }
            Self::PagedIndex => {
                assert_native_values(&file.dataset("keep").unwrap(), &[1i32, 2, 3, 4], order);
                let dataset = file.dataset("victim").unwrap();
                assert_native_values(&dataset, &(0..8192).collect::<Vec<i32>>(), order);
                assert_eq!(dataset.chunk(), Some(vec![512]));
                assert_eq!(dataset.space().unwrap().maxdims(), [None]);
            }
        }
        for name in self.dataset_names() {
            let dataset = file.dataset(&name).unwrap();
            if !matches!(self, Self::GenericLarge) {
                assert!(dataset.attr_names().unwrap().is_empty());
            }
            if name != "victim" {
                assert_eq!(dataset.layout(), NativeLayout::Contiguous);
                assert_eq!(dataset.chunk(), None);
            }
        }
        if let Some(persistence) = self.persistence() {
            let page = persistence.page_size();
            let properties = file.fcpl().unwrap();
            assert_eq!(
                properties.file_space_strategy(),
                FileSpaceStrategy::FreeSpaceManager {
                    paged: page.is_some(),
                    persist: true,
                    threshold: 1,
                }
            );
            if let Some(page) = page {
                assert_eq!(properties.file_space_page_size(), page);
            }
        }
        file.close().unwrap();
    }

    /// Checks hierarchy, numeric values, attribute maps, and storage properties through [`File`].
    ///
    /// Numeric datatypes must use `order`.
    ///
    /// # Panics
    ///
    /// Panics if metadata or values cannot be read, or any observation differs from the fixture.
    pub fn assert_contents(self, file: &File, order: NumericOrder) {
        let root = file.root();
        assert!(root.attrs().unwrap().is_empty());
        assert_eq!(
            root.groups().unwrap(),
            if matches!(self, Self::HugeLinks12) {
                vec!["g"]
            } else {
                vec![]
            }
        );
        assert_eq!(
            root.named_datatypes().unwrap(),
            if matches!(self, Self::CommittedDatatypeV1) {
                vec!["mytype"]
            } else {
                vec![]
            }
        );
        let mut datasets = root.datasets().unwrap();
        datasets.sort();
        let expected = if matches!(self, Self::HugeLinks12) {
            vec![]
        } else {
            self.dataset_names()
        };
        assert_eq!(datasets, expected);
        match self {
            Self::CommittedDatatypeV1 => {
                assert_eq!(
                    root.named_datatype("mytype").unwrap(),
                    integer_type(FixedPointType::i32(), order)
                );
                assert_eq!(root.named_datatype_references("mytype").unwrap(), 3);
                for name in ["first", "second"] {
                    assert_i32(file, name, &[1, 2], order);
                }
            }
            Self::GenericLarge => {
                let dataset = file.dataset("d").unwrap();
                assert_eq!(dataset.shape().unwrap(), [64]);
                assert_eq!(
                    dataset.datatype().unwrap(),
                    Datatype::FloatingPoint(FloatingPointType::ieee754_binary64(
                        order.pure_order()
                    ))
                );
                assert_eq!(dataset.read_f64().unwrap(), [1.0; 64]);
                assert_eq!(
                    dataset.attrs().unwrap(),
                    HashMap::from([("big_attr".to_owned(), AttrValue::I64Array(vec![7; 512]))])
                );
                assert_eq!(
                    dataset.attr_datatypes().unwrap(),
                    HashMap::from([(
                        "big_attr".to_owned(),
                        integer_type(FixedPointType::i64(), order)
                    )])
                );
            }
            Self::HardLinkUndo => {
                for name in ["aa", "bb"] {
                    assert_i32(file, name, &[1, 2, 3], order);
                }
            }
            Self::HugeLinks12 => {
                let group = file.group("g").unwrap();
                assert!(group.attrs().unwrap().is_empty());
                assert!(group.groups().unwrap().is_empty());
                assert!(group.named_datatypes().unwrap().is_empty());
                let mut names = group.datasets().unwrap();
                names.sort();
                let mut expected: Vec<_> = (0..12).map(huge_link_name).collect();
                expected.sort();
                assert_eq!(names, expected);
                for index in 0..12 {
                    assert_i32(
                        file,
                        &format!("g/{}", huge_link_name(index)),
                        &[index],
                        order,
                    );
                }
            }
            Self::PagedIndex => {
                assert_i32(file, "keep", &[1, 2, 3, 4], order);
                assert_i32(file, "victim", &(0..8192).collect::<Vec<_>>(), order);
                let dataset = file.dataset("victim").unwrap();
                assert_eq!(dataset.chunk_shape().unwrap(), Some(vec![512]));
                assert_eq!(
                    dataset.chunk_index().unwrap(),
                    Some(ChunkIndex::ExtensibleArray)
                );
            }
        }
        for name in self.dataset_names() {
            let dataset = file.dataset(&name).unwrap();
            assert!(dataset.filter_pipeline().unwrap().is_empty());
            if name != "victim" {
                assert_eq!(dataset.chunk_shape().unwrap(), None);
                assert!(matches!(
                    dataset.layout().unwrap(),
                    Layout::Contiguous { .. }
                ));
            }
        }
        if let Some(persistence) = self.persistence() {
            let page = persistence.page_size();
            let info = file.file_space_info().unwrap().unwrap();
            assert!(info.persist);
            assert_eq!(info.threshold, 1);
            assert_eq!(
                info.strategy,
                if page.is_some() {
                    PureStrategy::Page
                } else {
                    PureStrategy::FsmAggr
                }
            );
            if let Some(page) = page {
                assert_eq!(info.page_size.get(), page);
            }
        }
    }

    /// Checks the storage premise associated with the selected fixture.
    ///
    /// Pass the bytes from which `file` was opened. The checks cover datatype reference
    /// counts, continued headers, hard-link identity, huge heap objects, or chunk-index
    /// page separation, according to the fixture.
    ///
    /// # Panics
    ///
    /// Panics if metadata cannot be read, or the storage premise differs from the fixture.
    pub fn assert_representation(self, file: &File, bytes: &[u8]) {
        match self {
            Self::CommittedDatatypeV1 => {
                let links = symbol_table::Fixture::new(file, bytes)
                    .links(file.superblock().root_group_address);
                let address = links
                    .iter()
                    .find(|(name, _)| name == "mytype")
                    .unwrap()
                    .1
                    .get();
                let header = parse_header(file, bytes, address);
                assert_eq!(header.version, HEADER_VERSION_ONE);
                assert_eq!(header.reference_count, Some(3));
            }
            Self::GenericLarge => {
                let address = storage::root_object_address(file, bytes, "d");
                let spans = storage::header_extents(file.superblock(), bytes, address);
                assert!(spans.len() > 1);
                assert!(spans.iter().any(|(_, size)| *size > 512));
                let report = file.persisted_free_sections().unwrap().unwrap();
                let manager = report
                    .managers()
                    .iter()
                    .find(|manager| manager.role() == FreeSpaceManagerRole::PageLargeGeneric)
                    .unwrap();
                assert!(
                    manager
                        .free_sections()
                        .sections()
                        .iter()
                        .any(|section| section.address() % 512 != 0
                            || section.length().get() % 512 != 0)
                );
            }
            Self::HardLinkUndo => {
                let aa = storage::root_object_address(file, bytes, "aa");
                let bb = storage::root_object_address(file, bytes, "bb");
                assert_eq!(aa, bb);
            }
            Self::HugeLinks12 => {
                let address = storage::root_object_address(file, bytes, "g");
                let header = parse_header(file, bytes, address);
                let message = header
                    .messages
                    .iter()
                    .find(|message| message.msg_type == MessageType::LINK_INFO)
                    .unwrap();
                let info = LinkInfoMessage::parse_with_offset_width(
                    &message.data,
                    file.superblock().offset_size,
                )
                .unwrap();
                assert!(info.btree_name_index_address.is_some());
                let heap = FractalHeapHeader::parse(
                    bytes,
                    usize::try_from(info.fractal_heap_address.unwrap().get()).unwrap(),
                    file.superblock().offset_size.get(),
                    file.superblock().length_size.get(),
                )
                .unwrap();
                assert_eq!(heap.huge_objects_count, 12);
                assert_eq!(heap.managed_objects_count, 0);
            }
            Self::PagedIndex => assert_paged_index_storage(file, bytes),
        }
    }

    fn root_names(self) -> Vec<&'static str> {
        match self {
            Self::CommittedDatatypeV1 => vec!["first", "mytype", "second"],
            Self::GenericLarge => vec!["d"],
            Self::HardLinkUndo => vec!["aa", "bb"],
            Self::HugeLinks12 => vec!["g"],
            Self::PagedIndex => vec!["keep", "victim"],
        }
    }

    fn dataset_names(self) -> Vec<String> {
        match self {
            Self::CommittedDatatypeV1 => vec!["first".into(), "second".into()],
            Self::GenericLarge => vec!["d".into()],
            Self::HardLinkUndo => vec!["aa".into(), "bb".into()],
            Self::HugeLinks12 => (0..12)
                .map(|index| format!("g/{}", huge_link_name(index)))
                .collect(),
            Self::PagedIndex => vec!["keep".into(), "victim".into()],
        }
    }

    fn persistence(self) -> Option<Persistence> {
        match self {
            Self::GenericLarge => Some(Persistence::Paged(
                FileSpacePageSize::try_from(512).unwrap(),
            )),
            Self::HardLinkUndo => Some(Persistence::Aggregated),
            Self::PagedIndex => Some(Persistence::Paged(FileSpacePageSize::DEFAULT)),
            Self::CommittedDatatypeV1 | Self::HugeLinks12 => None,
        }
    }
}

enum Persistence {
    Aggregated,
    Paged(FileSpacePageSize),
}

impl Persistence {
    fn page_size(self) -> Option<u64> {
        match self {
            Self::Aggregated => None,
            Self::Paged(page) => Some(page.get()),
        }
    }
}

/// Writes twelve datasets under `g`, with link names longer than 5,000 bytes.
///
/// Each dataset contains its index as one [`i32`] value. The links occupy huge
/// fractal-heap objects in dense storage. Closes the file before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn huge_links_12(path: &Path) {
    const COUNT: usize = 12;
    let file = NativeFileBuilder::new()
        .with_fapl(|fapl| fapl.libver_latest())
        .create(path)
        .unwrap();
    let group = file.create_group("g").unwrap();
    for i in 0..COUNT {
        let name = format!("d{i}_{}", "x".repeat(5000));
        group
            .new_dataset::<i32>()
            .shape((1,))
            .create(name.as_str())
            .unwrap()
            .write(&[i as i32])
            .unwrap();
    }
    drop(group);
    file.close().unwrap();
}

/// Writes an unlimited `victim` dataset with 512-element chunks and a separate `keep` dataset.
///
/// `victim` contains [`i32`] values 0 through 8,191, and `keep` contains `[1, 2, 3, 4]`.
/// The file persists free-space managers with 4,096-byte pages. Closes the file
/// before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn paged_index(path: &Path) {
    let f = NativeFileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| {
            fcpl.file_space_strategy(persisting(true))
                .file_space_page_size(4096)
        })
        .create(path)
        .unwrap();
    let ds = f
        .new_dataset::<i32>()
        .shape(SimpleExtents::resizable(vec![8192]))
        .chunk((512,))
        .create("victim")
        .unwrap();
    ds.write_raw(&(0..8192i32).collect::<Vec<i32>>()).unwrap();
    f.new_dataset::<i32>()
        .shape((4,))
        .create("keep")
        .unwrap()
        .write_raw(&[1i32, 2, 3, 4])
        .unwrap();
    drop(ds);
    f.close().unwrap();
}

/// Writes a 512-element attribute on `d` in a file with persisted 512-byte pages.
///
/// The dataset contains 64 [`f64`] values equal to 1.0, and its attribute `big_attr`
/// contains 512 [`i64`] values equal to 7. Closes the file before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn generic_large(path: &Path) {
    let f = NativeFileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| {
            fcpl.file_space_strategy(persisting(true))
                .file_space_page_size(512)
        })
        .create(path)
        .unwrap();
    let ds = f.new_dataset::<f64>().shape((64,)).create("d").unwrap();
    ds.write_raw(&vec![1.0f64; 64]).unwrap();
    let a = ds
        .new_attr::<i64>()
        .shape((512,))
        .create("big_attr")
        .unwrap();
    a.write_raw(&vec![7i64; 512]).unwrap();
    drop(a);
    drop(ds);
    f.close().unwrap();
}

/// Writes `mytype` in a version 1 object header, shared by `first` and `second`.
///
/// The committed datatype is [`i32`], and both datasets contain `[1, 2]`.
/// Closes the file before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn committed_datatype_v1(path: &Path) {
    let f = NativeFileBuilder::new()
        .with_fapl(|fapl| fapl.libver_bounds(LibraryVersion::Earliest, LibraryVersion::V18))
        .create(path)
        .unwrap();
    let dtype = NativeDatatype::from_type::<i32>().unwrap();
    f.commit_datatype("mytype", &dtype).unwrap();
    for name in ["first", "second"] {
        f.new_dataset_builder()
            .empty_as(&dtype)
            .shape([2])
            .create(name)
            .unwrap()
            .write(&[1i32, 2])
            .unwrap();
    }
    drop(dtype);
    f.close().unwrap();
}

/// Writes `aa` and `bb` as two hard links to one [`i32`] dataset containing `[1, 2, 3]`.
///
/// The file persists free-space managers without paging. Closes the file before returning.
///
/// # Panics
///
/// Panics if native creation, writing, or close fails.
pub fn hard_link_undo(path: &Path) {
    let f = NativeFileBuilder::new()
        .with_fapl(|fapl| fapl.libver_v110())
        .with_fcpl(|fcpl| fcpl.file_space_strategy(persisting(false)))
        .create(path)
        .unwrap();
    f.new_dataset::<i32>()
        .shape((3,))
        .create("aa")
        .unwrap()
        .write(&[1i32, 2, 3])
        .unwrap();
    f.link_hard("aa", "bb").unwrap();
    f.close().unwrap();
}

fn persisting(paged: bool) -> FileSpaceStrategy {
    FileSpaceStrategy::FreeSpaceManager {
        paged,
        persist: true,
        threshold: 1,
    }
}

fn assert_native_values<T: H5Type + PartialEq + Debug>(
    dataset: &NativeDataset,
    values: &[T],
    order: NumericOrder,
) {
    assert_eq!(dataset.shape(), [values.len()]);
    assert_eq!(
        dataset.dtype().unwrap().to_descriptor().unwrap(),
        T::type_descriptor()
    );
    assert_eq!(dataset.dtype().unwrap().byte_order(), order.native_order());
    assert_eq!(dataset.read_raw::<T>().unwrap(), values);
    assert!(dataset.filters().is_empty());
}

fn assert_i32(file: &File, name: &str, values: &[i32], order: NumericOrder) {
    let dataset = file.dataset(name).unwrap();
    assert_eq!(dataset.shape().unwrap(), [values.len() as u64]);
    assert_eq!(
        dataset.datatype().unwrap(),
        integer_type(FixedPointType::i32(), order)
    );
    assert_eq!(dataset.read_i32().unwrap(), values);
    assert!(dataset.attrs().unwrap().is_empty());
}

fn integer_type(datatype: FixedPointType, order: NumericOrder) -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::from_layout(datatype.size().get(), order.pure_order(), datatype.layout())
            .unwrap(),
    )
}

fn huge_link_name(index: i32) -> String {
    format!("d{index}_{}", "x".repeat(5000))
}

fn widths(file: &File) -> FormatWidths {
    FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    }
}

fn parse_header(file: &File, bytes: &[u8], address: u64) -> ObjectHeader {
    ObjectHeader::parse_with_base_and_widths(
        bytes,
        AccessMode::ReadOnly,
        usize::try_from(address).unwrap(),
        widths(file),
        file.superblock().base_address,
    )
    .unwrap()
}

fn assert_paged_index_storage(file: &File, bytes: &[u8]) {
    let address = storage::root_object_address(file, bytes, "victim");
    let header = parse_header(file, bytes, address);
    let layout = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
        .unwrap();
    let DataLayout::Chunked {
        index: ChunkIndexLayout::ExtensibleArray {
            address: Some(address),
        },
        ..
    } = DataLayout::parse_with_widths(&layout.data, widths(file)).unwrap()
    else {
        panic!("expected an allocated extensible array")
    };
    let index = ExtensibleArrayHeader::parse_with_widths(
        bytes,
        usize::try_from(address.get()).unwrap(),
        widths(file),
    )
    .unwrap();
    assert_eq!(index.max_idx_set, 16);
    let metadata_pages = [address.get() / 4096, index.index_block_address.get() / 4096];
    let mut chunks = file.dataset("victim").unwrap().chunks().unwrap();
    chunks.sort_by(|left, right| left.offset.cmp(&right.offset));
    assert_eq!(
        chunks
            .iter()
            .map(|chunk| chunk.offset.clone())
            .collect::<Vec<_>>(),
        (0..16).map(|index| vec![index * 512]).collect::<Vec<_>>()
    );
    for chunk in chunks {
        assert_eq!(chunk.storage_size, 512 * 4);
        assert_eq!(chunk.filter_mask().raw_bits(), 0);
        for page in chunk.address / 4096..=(chunk.address + chunk.storage_size - 1) / 4096 {
            assert!(!metadata_pages.contains(&page));
        }
    }
}

// Section `subsubsec_fmt4_dataobject_hdr_prefix_one`, version 4.0.
const HEADER_VERSION_ONE: u8 = 1;
