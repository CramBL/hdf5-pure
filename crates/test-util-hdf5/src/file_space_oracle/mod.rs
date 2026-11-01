//! Compares persisted free-space sections with an independent libhdf5 enumeration.
//!
//! [`assert_persisted_sections`] compares buffered and streaming facade reports with the C
//! enumeration at base-relative stored addresses and checks the recorded manager contents.
//! [`FreeSpaceOracleCheckpoint`] records paired free-section reports, creation properties,
//! and file sizes.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::DecodedStoredAddress;
use hdf5_pure::FreeSections;
use hdf5_pure::FreeSpaceManagerRole;
use hdf5_pure::PersistedFreeSections;
use hdf5_pure::StoredAddress;
use hdf5_pure::{File, FileSpaceStrategy};
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::DEFAULT_THRESHOLD;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_sys::h5f::H5F_info2_t;
use hdf5_sys::{h5f, h5p};

use crate::file_space;

mod report;

/// Asserts persisted manager allocation sizes against file headers and a fresh C handle.
///
/// Returns the independently calculated on-disk total. Opens `path` read-only in each library
/// and checks that the file bytes remain unchanged.
///
/// The version 0 `FSHD` header has 18 fixed bytes, one address, and seven lengths, as defined
/// in "Free-space Index" of the [format specification, version 4.0][spec].
///
/// # Panics
///
/// Panics if file inspection fails, either library's count differs from the allocation sizes,
/// or the file bytes change during inspection.
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
pub fn assert_manager_metadata_bytes(path: &Path) -> u64 {
    let bytes = fs::read(path).unwrap();
    let reader = File::open(path).unwrap();
    let report = reader.persisted_free_sections().unwrap();
    let widths = FormatWidths {
        offsets: reader.superblock().offset_size,
        lengths: reader.superblock().length_size,
    };
    let header_bytes = 18 + u64::from(widths.offsets.get()) + 7 * u64::from(widths.lengths.get());
    let expected = report.as_ref().map_or(0, |report| {
        let expected = report
            .managers()
            .iter()
            .map(|manager| {
                let address =
                    usize::try_from(reader.resolve_address(manager.address()).unwrap().get())
                        .unwrap();
                let header = FreeSpaceManagerHeader::parse(widths, &bytes[address..]).unwrap();
                let allocated = if header.section_list_addr().is_undefined(widths.offsets) {
                    0
                } else {
                    header.section_list_allocated()
                };
                assert_eq!(
                    manager.metadata_bytes(),
                    header_bytes + allocated,
                    "{path:?}"
                );
                header_bytes + allocated
            })
            .sum::<u64>();
        assert_eq!(report.manager_metadata_bytes(), expected, "{path:?}");
        expected
    });
    reader.close().unwrap();
    let file = hdf5::File::open(path).unwrap();
    // `H5MF_get_freespace` opens and closes managers started for its query (`H5MF.c`, HDF5 1.14.6).
    // `H5FS_size` uses allocated bytes with the section list unloaded (`H5FS.c`, HDF5 1.14.6).
    assert_eq!(
        c_manager_metadata_bytes(&file).unwrap(),
        expected,
        "fresh handle; {path:?}"
    );
    file.close().unwrap();
    assert_eq!(
        fs::read(path).unwrap(),
        bytes,
        "metadata inspection; {path:?}"
    );
    expected
}

fn c_manager_metadata_bytes(file: &hdf5::File) -> hdf5::Result<u64> {
    hdf5::sync::sync(|| {
        let mut info = H5F_info2_t::default();
        // SAFETY: The library lock serializes access. The file owns a valid identifier, and info
        // is writable storage of the type `H5Fget_info2` requires for the duration of the call.
        hdf5::h5check(unsafe { h5f::H5Fget_info2(file.id(), &mut info) })?;
        Ok(info.free.meta_size)
    })
}

/// Checks persisted sections, creation properties, and file sizes through both libraries.
///
/// The caller must close all writing sessions before observation. `case` and `step` identify the
/// workload and checkpoint in the returned report and diagnostics.
///
/// # Panics
///
/// Panics if either library reports an error, metadata is invalid, observations disagree, or
/// the file bytes change during observation.
pub fn observe(path: &Path, case: &str, step: &str) -> FreeSpaceOracleCheckpoint {
    assert_persisted_sections(path);
    let observation = FreeSpaceOracleCheckpoint::observe(path, case, step)
        .unwrap_or_else(|error| panic!("{case}/{step}: {error:?}"));
    observation.assert_equal();
    observation
}

/// Checks supported persisted free-section inspection against libhdf5 for a closed file.
///
/// The caller closes every writing session before calling this function. Compares buffered
/// and streaming [`File::persisted_free_sections`] reports with `H5Fget_free_sections` and
/// `H5Fget_freespace`. Checks each manager against its recorded slot and encoded contents,
/// verifies combined and per-manager totals, and compares the complete file bytes before
/// and after observation. Runs without collecting ownership accounting.
///
/// # Panics
///
/// Panics if either library reports an error, metadata cannot be read or parsed, reports or
/// manager records disagree, or observation changes the file bytes.
#[track_caller]
pub fn assert_persisted_sections(path: &Path) {
    let bytes = fs::read(path).unwrap();
    let reader = File::open(path).unwrap();
    let report = reader.persisted_free_sections().unwrap();
    let streamed = File::open_streaming(path)
        .unwrap()
        .persisted_free_sections()
        .unwrap();
    assert_eq!(report, streamed, "buffered/streamed sections; {path:?}");
    let configured = reader
        .file_space_info()
        .unwrap()
        .is_some_and(|info| info.persist);
    assert_eq!(report.is_some(), configured, "persistence; {path:?}");
    let boundary = reader.superblock().eof_address - reader.superblock().base_address.get();
    let rust = report
        .as_ref()
        .map(|report| section_pairs(report.free_sections()))
        .unwrap_or_default();
    if let Some(report) = &report {
        assert_manager_sections(&reader, report, &bytes);
    }
    reader.close().unwrap();
    let file = hdf5::File::open(path).unwrap();
    let c = SectionMap::new(
        "libhdf5",
        file_space::libhdf5_free_sections(&file).unwrap(),
        boundary,
    )
    .unwrap();
    let rust = SectionMap::new("supported API", rust, boundary).unwrap();
    rust.compare(&c)
        .unwrap_or_else(|error| panic!("supported all-manager sections; {path:?}: {error:?}"));
    let (c_bytes, _) = c_sizes(&file).unwrap();
    assert_eq!(rust.bytes(), c_bytes, "H5Fget_freespace; {path:?}");
    if let Some(report) = report {
        let all = report.free_sections();
        assert_eq!(all.section_count(), c.sections().len(), "count; {path:?}");
        assert_eq!(all.free_bytes(), c.bytes(), "section bytes; {path:?}");
        assert_eq!(
            all.section_count(),
            report
                .managers()
                .iter()
                .map(|manager| manager.free_sections().section_count())
                .sum::<usize>(),
            "manager section counts; {path:?}"
        );
        assert_eq!(
            all.free_bytes(),
            report
                .managers()
                .iter()
                .map(|manager| manager.free_sections().free_bytes())
                .sum::<u64>(),
            "manager byte totals; {path:?}"
        );
    }
    file.close().unwrap();
    assert!(
        bytes == fs::read(path).unwrap(),
        "observation changed bytes; {path:?}"
    );
}

/// Checks reported manager slots, roles, counts, bytes, and sections against encoded metadata.
fn assert_manager_sections(file: &File, report: &PersistedFreeSections, bytes: &[u8]) {
    let info = file.file_space_info().unwrap().unwrap();
    let defined = info
        .manager_addrs
        .iter()
        .enumerate()
        .filter_map(|(slot, &address)| match address {
            DecodedStoredAddress::Defined(address) => Some((slot, address)),
            DecodedStoredAddress::Undefined => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        report
            .managers()
            .iter()
            .map(|manager| (manager.slot(), manager.address()))
            .collect::<Vec<_>>(),
        defined
    );
    let widths = FormatWidths {
        offsets: file.superblock().offset_size,
        lengths: file.superblock().length_size,
    };
    for manager in report.managers() {
        let expected_role = match info.strategy {
            FileSpaceStrategy::FsmAggr => FSM_AGGR_ROLES.get(manager.slot()),
            FileSpaceStrategy::Page => PAGE_ROLES.get(manager.slot()),
            FileSpaceStrategy::Aggr | FileSpaceStrategy::None => None,
        }
        .copied()
        .unwrap_or(FreeSpaceManagerRole::Unknown);
        assert_eq!(manager.role(), expected_role, "slot {}", manager.slot());
        let address =
            usize::try_from(file.resolve_address(manager.address()).unwrap().get()).unwrap();
        let header = FreeSpaceManagerHeader::parse(widths, &bytes[address..]).unwrap();
        let sections = manager.free_sections();
        assert_eq!(
            u64::try_from(sections.section_count()).unwrap(),
            header.total_sections()
        );
        assert_eq!(sections.free_bytes(), header.total_space());
        if header.total_sections() == 0 {
            assert_eq!(sections.sections(), []);
            continue;
        }
        let start = usize::try_from(
            file.resolve_address(header.section_list_addr())
                .unwrap()
                .get(),
        )
        .unwrap();
        let used = usize::try_from(header.section_list_used()).unwrap();
        let encoded = format::parse_section_info(
            widths,
            &bytes[start..start + used],
            manager.address(),
            &header,
        )
        .unwrap();
        let mut encoded = encoded
            .iter()
            .map(|section| (section.addr.get(), section.size))
            .collect::<Vec<_>>();
        encoded.sort_unstable();
        assert_eq!(
            section_pairs(sections),
            encoded,
            "encoded manager slot {}",
            manager.slot()
        );
    }
}

/// Checks stored coordinates and returns address-length pairs for comparison.
fn section_pairs(sections: &FreeSections<StoredAddress>) -> Vec<(u64, u64)> {
    sections
        .sections()
        .iter()
        .map(|section| {
            let _: StoredAddress = section.address();
            (section.address().get(), section.length().get())
        })
        .collect()
}

/// The persisted free-space maps, creation properties, and file sizes for one closed file.
#[derive(Debug)]
pub struct FreeSpaceOracleCheckpoint {
    /// The workload and creation settings used to identify the file.
    pub case: String,
    /// The close or reopen checkpoint within the workload.
    pub step: String,
    /// The sections decoded by [`File::persisted_free_sections`].
    pub rust: SectionMap,
    /// The sections enumerated by libhdf5 through `H5Fget_free_sections`.
    pub c: SectionMap,
    /// The byte total reported separately by `H5Fget_freespace`.
    pub c_reported_freespace: u64,
    /// The absolute end-of-file position stored in the superblock, in bytes.
    pub rust_logical_size: u64,
    /// The file length reported by `H5Fget_filesize`, including the userblock.
    pub c_file_size: u64,
    /// The physical file length in bytes.
    pub physical_file_size: u64,
    /// The base address at the end of the userblock, or zero.
    pub base: u64,
    properties: CreationProperties,
}

impl FreeSpaceOracleCheckpoint {
    /// Opens a closed file with each library and records its persisted sections.
    ///
    /// The caller must close all writing sessions before calling this method. The observer opens
    /// the file read-only with each library and compares creation properties. It compares the
    /// complete file bytes before and after observation.
    ///
    /// # Errors
    ///
    /// Returns [`OracleError::Io`], [`OracleError::Rust`], or [`OracleError::C`] if an operation
    /// fails. Returns [`OracleError::Allocation`] if the section buffer cannot be allocated,
    /// [`OracleError::Enumeration`] if the enumeration count changes, or [`OracleError::Extent`]
    /// if either section map is invalid. Returns [`OracleError::Properties`] if creation settings
    /// disagree, [`OracleError::Size`] if a size is inconsistent with the base or physical length,
    /// or [`OracleError::ChangedBytes`] if observation changes the file.
    pub fn observe(path: &Path, case: &str, step: &str) -> Result<Self, OracleError> {
        let before = fs::read(path).map_err(OracleError::Io)?;
        let reader = File::open(path).map_err(OracleError::Rust)?;
        let base = reader.superblock().base_address.get();
        let rust_logical_size = reader.superblock().eof_address;
        let physical_file_size =
            u64::try_from(before.len()).map_err(|_error| OracleError::Size {
                base,
                rust: rust_logical_size,
                c: 0,
                physical: u64::MAX,
            })?;
        let boundary = rust_logical_size
            .checked_sub(base)
            .ok_or(OracleError::Size {
                base,
                rust: rust_logical_size,
                c: 0,
                physical: physical_file_size,
            })?;
        let rust = SectionMap::new(
            "hdf5-pure",
            reader
                .persisted_free_sections()
                .map_err(OracleError::Rust)?
                .as_ref()
                .map(|report| section_pairs(report.free_sections()))
                .unwrap_or_default(),
            boundary,
        )?;
        let info = reader.file_space_info().map_err(OracleError::Rust)?;
        reader.close().map_err(OracleError::Rust)?;
        let file = hdf5::File::open(path).map_err(|source| OracleError::C {
            operation: "H5Fopen",
            source,
        })?;
        let c = SectionMap::new(
            "libhdf5",
            file_space::libhdf5_free_sections(&file)?,
            boundary,
        )?;
        let (c_reported_freespace, c_file_size) = c_sizes(&file)?;
        let properties = file.create_plist().map_err(|source| OracleError::C {
            operation: "H5Fget_create_plist",
            source,
        })?;
        let c_strategy = properties
            .get_file_space_strategy()
            .map_err(|source| OracleError::C {
                operation: "H5Pget_file_space_strategy",
                source,
            })?;
        let page = properties
            .get_file_space_page_size()
            .map_err(|source| OracleError::C {
                operation: "H5Pget_file_space_page_size",
                source,
            })?;
        let c_base = properties
            .get_userblock()
            .map_err(|source| OracleError::C {
                operation: "H5Pget_userblock",
                source,
            })?;
        let (c_persist, c_threshold) = hdf5::sync::sync(|| {
            let mut strategy = h5f::H5F_FSPACE_STRATEGY_FSM_AGGR;
            let mut persist = 0;
            let mut threshold = 0;
            // SAFETY: The property list remains live, the outputs are writable and the library
            // lock serializes access to the C library.
            if unsafe {
                h5p::H5Pget_file_space_strategy(
                    properties.id(),
                    &mut strategy,
                    &mut persist,
                    &mut threshold,
                )
            } < 0
            {
                return Err(c_error("H5Pget_file_space_strategy"));
            }
            Ok((persist != 0, threshold))
        })?;
        let expected = match info
            .as_ref()
            .map(|i| i.strategy)
            .unwrap_or(FileSpaceStrategy::FsmAggr)
        {
            FileSpaceStrategy::FsmAggr | FileSpaceStrategy::Page => CStrategy::FreeSpaceManager {
                paged: info
                    .as_ref()
                    .is_some_and(|i| i.strategy == FileSpaceStrategy::Page),
                persist: info.as_ref().is_some_and(|i| i.persist),
                threshold: info.as_ref().map_or(DEFAULT_THRESHOLD, |i| i.threshold),
            },
            FileSpaceStrategy::Aggr => CStrategy::PageAggregation,
            FileSpaceStrategy::None => CStrategy::None,
        };
        if info
            .as_ref()
            .is_some_and(|i| i.persist != c_persist || i.threshold != c_threshold)
            || expected != c_strategy
            || c_base != base
            || info.as_ref().is_some_and(|i| i.page_size.get() != page)
        {
            return Err(OracleError::Properties {
                rust: format!(
                    "{expected:?}/page={:?}/base={base}",
                    info.as_ref().map(|i| i.page_size.get())
                ),
                c: format!(
                    "{c_strategy:?}/persist={c_persist}/threshold={c_threshold}/page={page}/base={c_base}"
                ),
            });
        }
        let properties = CreationProperties {
            rust: PropertyValues {
                strategy: strategy_name(&expected),
                persist: info.as_ref().is_some_and(|i| i.persist),
                threshold: info.as_ref().map_or(DEFAULT_THRESHOLD, |i| i.threshold),
                page_size: info.as_ref().map(|i| i.page_size.get()),
                base,
            },
            c: PropertyValues {
                strategy: strategy_name(&c_strategy),
                persist: c_persist,
                threshold: c_threshold,
                page_size: Some(page),
                base: c_base,
            },
        };
        file.close().map_err(|source| OracleError::C {
            operation: "H5Fclose",
            source,
        })?;
        if c_file_size != physical_file_size || rust_logical_size > physical_file_size {
            return Err(OracleError::Size {
                base,
                rust: rust_logical_size,
                c: c_file_size,
                physical: physical_file_size,
            });
        }
        if before != fs::read(path).map_err(OracleError::Io)? {
            return Err(OracleError::ChangedBytes);
        }
        Ok(Self {
            case: case.to_owned(),
            step: step.to_owned(),
            rust,
            c,
            c_reported_freespace,
            rust_logical_size,
            c_file_size,
            physical_file_size,
            base,
            properties,
        })
    }

    /// Requires exact section boundaries and equal C section and free-space totals.
    ///
    /// On a map mismatch, prints the first differing index, complete missing and extra section
    /// lists, and the paired report. The total-byte assertion applies to settled read-only
    /// checkpoints.
    ///
    /// # Panics
    ///
    /// Panics if the maps differ or `H5Fget_freespace` differs from the C section-byte sum.
    #[track_caller]
    pub fn assert_equal(&self) {
        if let Err(error) = self.rust.compare(&self.c) {
            panic!("{}/{}, {error:?}\n{}", self.case, self.step, self.render());
        }
        assert_eq!(
            self.c_reported_freespace, self.c.bytes,
            "settled read-only free-space total; {}/{}",
            self.case, self.step
        );
    }
}

/// The file-space creation properties observed through both libraries.
#[derive(Debug)]
struct CreationProperties {
    rust: PropertyValues,
    c: PropertyValues,
}

/// File-space settings from one library, with an optional page size.
#[derive(Debug)]
struct PropertyValues {
    strategy: &'static str,
    persist: bool,
    threshold: u64,
    page_size: Option<u64>,
    base: u64,
}

/// A sorted list of positive, disjoint sections within a base-relative allocation boundary.
///
/// Adjacent sections retain their individual boundaries.
#[derive(Debug)]
pub struct SectionMap {
    sections: Vec<(u64, u64)>,
    bytes: u64,
}

impl SectionMap {
    /// Sorts sections by address and length while preserving each section boundary.
    ///
    /// Each pair contains a base-relative stored address and a byte length. `boundary` is the
    /// allocation end in the same coordinate system. `source` identifies the enumerator in errors.
    ///
    /// # Errors
    ///
    /// Returns [`OracleError::Extent`] if a section is empty, overlaps another section, extends
    /// beyond `boundary`, or overflows an address or byte total. Duplicate sections overlap.
    pub fn new(
        source: &'static str,
        mut sections: Vec<(u64, u64)>,
        boundary: u64,
    ) -> Result<Self, OracleError> {
        sections.sort_unstable();
        let mut end = 0;
        let mut bytes = 0u64;
        for (index, &(address, length)) in sections.iter().enumerate() {
            let fail = |reason| OracleError::Extent {
                source,
                index,
                address,
                length,
                boundary,
                reason,
            };
            if length == 0 {
                return Err(fail(ExtentError::ZeroLength));
            }
            let next = address
                .checked_add(length)
                .ok_or_else(|| fail(ExtentError::AddressOverflow))?;
            if address < end {
                return Err(fail(ExtentError::Overlap));
            }
            if next > boundary {
                return Err(fail(ExtentError::OutOfBounds));
            }
            bytes = bytes
                .checked_add(length)
                .ok_or_else(|| fail(ExtentError::ByteSumOverflow))?;
            end = next;
        }
        Ok(Self { sections, bytes })
    }

    /// Returns the complete sections in ascending address and length order.
    pub fn sections(&self) -> &[(u64, u64)] {
        &self.sections
    }

    /// Returns the checked sum of the section lengths in bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Compares exact section boundaries with the reference library's map.
    ///
    /// # Errors
    ///
    /// Returns [`OracleError::Mismatch`] if either an address or a length differs. The error
    /// includes the first unequal index and every missing or extra section relative to `c`.
    pub fn compare(&self, c: &Self) -> Result<(), OracleError> {
        if self.sections == c.sections {
            return Ok(());
        }
        let rust_set = self.sections.iter().copied().collect::<BTreeSet<_>>();
        let c_set = c.sections.iter().copied().collect::<BTreeSet<_>>();
        Err(OracleError::Mismatch {
            first: self
                .sections
                .iter()
                .zip(&c.sections)
                .position(|(a, b)| a != b)
                .unwrap_or(self.sections.len().min(c.sections.len())),
            missing: c_set.difference(&rust_set).copied().collect(),
            extra: rust_set.difference(&c_set).copied().collect(),
        })
    }
}

/// An observation failure or an exact section-map mismatch.
#[derive(Debug)]
pub enum OracleError {
    /// The C section buffer could not be allocated.
    Allocation {
        /// The number of section records in the failed allocation.
        count: usize,
    },
    /// A C API call failed.
    C {
        operation: &'static str,
        source: hdf5::Error,
    },
    /// The file bytes differed after observation.
    ChangedBytes,
    /// The enumeration returned a different section count from its initial count query.
    Enumeration {
        /// The count from the initial buffer-size query.
        expected: usize,
        /// The count from the record enumeration.
        actual: usize,
    },
    /// A section map contained an invalid extent.
    Extent {
        /// The enumerator that supplied the section.
        source: &'static str,
        /// The section's index after sorting.
        index: usize,
        address: u64,
        length: u64,
        /// The base-relative end of allocated storage.
        boundary: u64,
        reason: ExtentError,
    },
    /// A filesystem operation failed.
    Io(std::io::Error),
    /// The Rust and C maps differed at a section boundary.
    Mismatch {
        /// The first unequal index in sorted order, or the first absent row.
        first: usize,
        /// The reference sections absent from the Rust map.
        missing: Vec<(u64, u64)>,
        /// The Rust sections absent from the reference map.
        extra: Vec<(u64, u64)>,
    },
    /// The libraries reported different creation properties.
    Properties {
        /// The settings decoded by Rust.
        rust: String,
        /// The settings returned by the C library.
        c: String,
    },
    /// A Rust file operation failed.
    Rust(hdf5_pure::Error),
    /// A reported size was inconsistent with the base address or physical file length.
    Size {
        base: u64,
        rust: u64,
        c: u64,
        physical: u64,
    },
}

/// The invalid address, length, or relationship between adjacent sorted sections.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExtentError {
    /// The section end exceeds the address range.
    AddressOverflow,
    /// The sum of the section lengths exceeds the byte-count range.
    ByteSumOverflow,
    /// The section extends beyond the allocation boundary.
    OutOfBounds,
    /// The section overlaps its predecessor, including an identical section.
    Overlap,
    /// The section has no bytes.
    ZeroLength,
}

/// Requires the second enumeration call to return the initially reported section count.
///
/// # Errors
///
/// Returns [`OracleError::C`] if `actual` reports a C API failure, or
/// [`OracleError::Enumeration`] if the nonnegative count differs from `expected`.
pub fn enumeration_complete(expected: usize, actual: isize) -> Result<(), OracleError> {
    let actual = enumeration_count(actual)?;
    if expected != actual {
        return Err(OracleError::Enumeration { expected, actual });
    }
    Ok(())
}

/// Converts a C section count while preserving an API failure as an error.
///
/// The caller must hold the C library lock while obtaining and converting the count, so the C
/// error stack belongs to the failed `H5Fget_free_sections` call.
///
/// # Errors
///
/// Returns [`OracleError::C`] if `count` is negative or cannot fit in [`usize`].
pub fn enumeration_count(count: isize) -> Result<usize, OracleError> {
    usize::try_from(count).map_err(|_error| c_error("H5Fget_free_sections"))
}

fn strategy_name(strategy: &CStrategy) -> &'static str {
    match strategy {
        CStrategy::FreeSpaceManager { paged: true, .. } => "PAGE",
        CStrategy::FreeSpaceManager { paged: false, .. } => "FSM_AGGR",
        CStrategy::PageAggregation => "AGGR",
        CStrategy::None => "NONE",
    }
}

/// Returns the C free-space total followed by the physical file length, in bytes.
///
/// Calls `H5Fget_freespace` and `H5Fget_filesize` independently of section enumeration.
///
/// # Errors
///
/// Returns [`OracleError::C`] if either C call fails.
fn c_sizes(file: &hdf5::File) -> Result<(u64, u64), OracleError> {
    hdf5::sync::sync(|| {
        // SAFETY: The library lock serializes access and `file` retains a live file identifier.
        let free = unsafe { h5f::H5Fget_freespace(file.id()) };
        let free = u64::try_from(free).map_err(|_error| c_error("H5Fget_freespace"))?;
        let mut size = 0;
        // SAFETY: The library lock serializes access, the file is live and `size` is writable.
        if unsafe { h5f::H5Fget_filesize(file.id(), &mut size) } < 0 {
            return Err(c_error("H5Fget_filesize"));
        }
        Ok((free, size))
    })
}

fn c_error(operation: &'static str) -> OracleError {
    OracleError::C {
        operation,
        source: hdf5::Error::query().unwrap_or_else(|source| source),
    }
}

// `H5O__fsinfo_decode` in `H5Ofsinfo.c`, HDF5 2.2.0, maps version 0 slots to managers
// without paging.
const FSM_AGGR_ROLES: [FreeSpaceManagerRole; 6] = [FreeSpaceManagerRole::FsmAggr; 6];
// `H5F_mem_page_t` in `H5Fprivate.h`, HDF5 2.2.0, assigns metadata, raw, and generic indices
// 1, 3, and 7.
// `H5O__fsinfo_encode` in `H5Ofsinfo.c`, HDF5 2.2.0, stores each index minus one.
const PAGE_ROLES: [FreeSpaceManagerRole; 12] = [
    FreeSpaceManagerRole::PageSmallMetadata,
    FreeSpaceManagerRole::Unknown,
    FreeSpaceManagerRole::PageSmallRawData,
    FreeSpaceManagerRole::Unknown,
    FreeSpaceManagerRole::Unknown,
    FreeSpaceManagerRole::Unknown,
    FreeSpaceManagerRole::PageLargeGeneric,
    FreeSpaceManagerRole::Unknown,
    FreeSpaceManagerRole::Unknown,
    FreeSpaceManagerRole::Unknown,
    FreeSpaceManagerRole::Unknown,
    FreeSpaceManagerRole::Unknown,
];
