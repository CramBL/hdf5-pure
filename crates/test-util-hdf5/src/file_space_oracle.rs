//! Compares persisted free-space sections with an independent libhdf5 enumeration.
//!
//! [`assert_persisted_sections`] compares buffered and streaming facade reports with the C
//! enumeration at base-relative stored addresses and checks the recorded manager contents.
//! [`FreeSpaceOracleCheckpoint`] records paired free-section reports, creation properties,
//! and file sizes.

use std::collections::BTreeSet;
use std::fmt::Display;
use std::fs;
use std::path::Path;

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::DecodedStoredAddress;
use hdf5_pure::FreeSectionCoordinates;
use hdf5_pure::FreeSections;
use hdf5_pure::FreeSpaceManagerRole;
use hdf5_pure::PersistedFreeSections;
use hdf5_pure::{File, FileSpaceStrategy};
use hdf5_pure_core::__private::BaseAddressExt;
use hdf5_pure_format::__private as format;
use hdf5_pure_format::__private::DEFAULT_THRESHOLD;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::FreeSpaceManagerHeader;
use hdf5_sys::{h5f, h5p};

use crate::file_space;

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
    let mut legacy = reader.persisted_free_space().unwrap();
    legacy.sort_unstable();
    assert_eq!(rust, legacy, "tuple compatibility; {path:?}");
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
    let base = file.superblock().base_address;
    for manager in report.managers() {
        let expected_role = match info.strategy {
            FileSpaceStrategy::FsmAggr => FSM_AGGR_ROLES.get(manager.slot()),
            FileSpaceStrategy::Page => PAGE_ROLES.get(manager.slot()),
            FileSpaceStrategy::Aggr | FileSpaceStrategy::None => None,
        }
        .copied()
        .unwrap_or(FreeSpaceManagerRole::Unknown);
        assert_eq!(manager.role(), expected_role, "slot {}", manager.slot());
        let address = usize::try_from(base.absolute(manager.address()).unwrap()).unwrap();
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
        let start = usize::try_from(base.absolute(header.section_list_addr()).unwrap()).unwrap();
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
fn section_pairs(sections: &FreeSections) -> Vec<(u64, u64)> {
    sections
        .sections()
        .iter()
        .map(|section| {
            assert_eq!(section.coordinates(), FreeSectionCoordinates::Stored);
            (section.address(), section.length().get())
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
    /// The sections decoded by [`File::persisted_free_space`].
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
    properties: String,
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
            reader.persisted_free_space().map_err(OracleError::Rust)?,
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
        let mut properties = String::from("file-space properties\n");
        for row in [
            paired_row("property", "hdf5-pure", "libhdf5"),
            paired_row(
                "strategy",
                strategy_name(&expected),
                strategy_name(&c_strategy),
            ),
            paired_row(
                "persistence",
                info.as_ref().is_some_and(|i| i.persist),
                c_persist,
            ),
            paired_row(
                "threshold (bytes)",
                info.as_ref().map_or(DEFAULT_THRESHOLD, |i| i.threshold),
                c_threshold,
            ),
            paired_row(
                "page size (bytes)",
                info.as_ref().map_or_else(
                    || "not stored".to_owned(),
                    |i| i.page_size.get().to_string(),
                ),
                page,
            ),
            paired_row("base / userblock (bytes)", base, c_base),
        ] {
            properties.push_str(&row);
        }
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

    /// Formats every section from both libraries beside file sizes and free-space totals.
    ///
    /// Missing rows and unequal pairs are marked explicitly. Section addresses and lengths are
    /// formatted in aligned hexadecimal columns padded with zeros. Reports include both
    /// section-byte sums and the separate `H5Fget_freespace` total.
    pub fn render(&self) -> String {
        let mut out = format!(
            "case:       {}\ncheckpoint: {}\n\n{}\n",
            self.case, self.step, self.properties
        );
        self.render_sizes(&mut out);
        self.render_sections(&mut out);
        self.render_totals(&mut out);
        out.push_str(&format!(
            "\nexact section map equal: {}\n\n",
            self.rust.sections == self.c.sections
        ));
        out
    }

    fn render_sizes(&self, out: &mut String) {
        out.push_str("file sizes (absolute bytes)\n");
        out.push_str(&format!("  {:<28} {:>14}\n", "measurement", "bytes"));
        for (label, bytes) in [
            ("hdf5-pure stored EOF", self.rust_logical_size),
            ("libhdf5 H5Fget_filesize", self.c_file_size),
            ("physical file", self.physical_file_size),
        ] {
            out.push_str(&format!("  {label:<28} {bytes:>14}\n"));
        }
        out.push('\n');
    }

    fn render_sections(&self, out: &mut String) {
        let count = self.rust.sections.len().max(self.c.sections.len());
        let (max_address, max_length) = self
            .rust
            .sections
            .iter()
            .chain(&self.c.sections)
            .fold((0, 0), |(address, length), &(at, len)| {
                (address.max(at), length.max(len))
            });
        let index_width = count.to_string().len().max("index".len());
        let address_width = format!("{max_address:#x}").len().max(10);
        let length_width = format!("{max_length:#x}").len().max(10);
        let section_width = address_width + length_width + 4;
        let columns = format!(
            " {:>address_width$}  {:>length_width$} ",
            "address", "length"
        );
        out.push_str("free-space sections (stored addresses, byte lengths)\n");
        out.push_str(&format!(
            "  {:>index_width$} | {:^section_width$} | {:^section_width$} | match\n",
            "index", "hdf5-pure", "libhdf5"
        ));
        out.push_str(&format!(
            "  {:>index_width$} | {columns} | {columns} |\n",
            ""
        ));
        out.push_str(&format!(
            "  {}-+-{}-+-{}-+------\n",
            "-".repeat(index_width),
            "-".repeat(section_width),
            "-".repeat(section_width)
        ));
        if count == 0 {
            out.push_str("  (no sections)\n");
        }
        for index in 0..count {
            let rust = self.rust.sections.get(index);
            let c = self.c.sections.get(index);
            let row = |section: Option<&(u64, u64)>| match section {
                Some((address, length)) => {
                    format!("({address:#0address_width$x}, {length:#0length_width$x})")
                }
                None => format!("{:<section_width$}", "<missing>"),
            };
            out.push_str(&format!(
                "  {index:>index_width$} | {} | {} | {}\n",
                row(rust),
                row(c),
                if rust == c { "yes" } else { "NO" }
            ));
        }
        out.push('\n');
    }

    fn render_totals(&self, out: &mut String) {
        out.push_str("free-space totals\n");
        for row in [
            paired_row("measurement", "hdf5-pure", "libhdf5"),
            paired_row(
                "section count",
                self.rust.sections.len(),
                self.c.sections.len(),
            ),
            paired_row("section bytes", self.rust.bytes, self.c.bytes),
            paired_row("H5Fget_freespace", "-", self.c_reported_freespace),
            paired_row(
                "reported - section bytes",
                "-",
                i128::from(self.c_reported_freespace) - i128::from(self.c.bytes),
            ),
        ] {
            out.push_str(&row);
        }
        out.push('\n');
    }
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

/// Formats a label and two values in the report's aligned columns.
fn paired_row(label: &str, left: impl Display, right: impl Display) -> String {
    format!("  {label:<28} {left:>14} {right:>14}\n")
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
