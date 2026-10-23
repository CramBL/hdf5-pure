//! Compares persisted free-space sections with an independent libhdf5 enumeration.
//!
//! The observer opens the same closed file with each library and compares section boundaries at
//! base-relative stored addresses. It also records Rust ownership accounting for files the editor
//! supports. The C enumeration covers free-space-manager state. Live raw data and metadata are
//! classified only by Rust's [`File::__space_ownership`] diagnostic.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use hdf5::plist::file_create::FileSpaceStrategy as CStrategy;
use hdf5_pure::{File, FileAccessProperties, FileSpaceStrategy, MemoryStrategy, SpaceOwnership};
use hdf5_pure_core::{LengthWidth, OffsetWidth};
use hdf5_pure_format::__private::DEFAULT_THRESHOLD;
use hdf5_sys::{h5f, h5p};

use crate::file_space;

/// The persisted free-space maps, file sizes, and ownership accounting for one closed file.
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
    /// The Rust ownership diagnostic, absent when either format width is smaller than eight bytes.
    pub ownership: Option<SpaceOwnership>,
    properties: String,
}

impl FreeSpaceOracleCheckpoint {
    /// Opens a closed file with each library and records its persisted sections.
    ///
    /// The caller must close all writing sessions before calling this method. The observer opens
    /// the file read-only with each library, compares creation properties, and obtains Rust
    /// ownership accounting through an unmodified mirrored editing handle. It compares the
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
        let info = reader.file_space_info().cloned();
        let editable = reader.superblock().offset_size == OffsetWidth::Eight
            && reader.superblock().length_size == LengthWidth::Eight;
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
        let properties = format!(
            "rust={expected:?} libhdf5={c_strategy:?} c_persist={c_persist} c_threshold={c_threshold} page={page} base={base}"
        );
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
        let ownership = if editable {
            let editor = File::open_rw_with_options(
                path,
                FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Mirrored),
            )
            .map_err(OracleError::Rust)?;
            let ownership = editor.__space_ownership().map_err(OracleError::Rust)?;
            editor.close().map_err(OracleError::Rust)?;
            Some(ownership)
        } else {
            None
        };
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
            ownership,
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

    /// Formats every section from both libraries beside file sizes and Rust ownership totals.
    ///
    /// Missing rows and unequal pairs are marked explicitly. Reports include both section-byte
    /// sums and the separate `H5Fget_freespace` total.
    pub fn render(&self) -> String {
        let mut out = format!(
            "checkpoint: {}/{}\nproperties: {}\nfile: rust_eof={} libhdf5_filesize={} physical={} base={}\nfree sections (base-relative address, length)\n  index  hdf5-pure                         libhdf5                           match\n",
            self.case,
            self.step,
            self.properties,
            self.rust_logical_size,
            self.c_file_size,
            self.physical_file_size,
            self.base
        );
        for index in 0..self.rust.sections.len().max(self.c.sections.len()) {
            let rust = self.rust.sections.get(index);
            let c = self.c.sections.get(index);
            let row = |section: Option<&(u64, u64)>| match section {
                Some((address, length)) => format!("({address:#x}, {length:#x})"),
                None => "<missing>".to_owned(),
            };
            out.push_str(&format!(
                "  {index:<6} {:<33} {:<33} {}\n",
                row(rust),
                row(c),
                if rust == c { "yes" } else { "NO" }
            ));
        }
        out.push_str(&format!("free: rust_sections={} c_sections={} rust_bytes={} c_bytes={}\nlibhdf5_H5Fget_freespace={} difference_from_c_sections={}\n",
            self.rust.sections.len(), self.c.sections.len(), self.rust.bytes, self.c.bytes,
            self.c_reported_freespace, i128::from(self.c_reported_freespace) - i128::from(self.c.bytes)));
        if let Some(ownership) = &self.ownership {
            out.push_str(&format!(
                "ownership (Rust): analysis_complete={}\n",
                ownership.analysis_complete
            ));
            for (label, sections) in [
                ("userblock", &ownership.userblock_space),
                ("live_raw", &ownership.live_raw_space),
                ("live_metadata", &ownership.live_metadata_space),
                ("managers", &ownership.free_space_manager_space),
                ("reusable", &ownership.reusable_free_space),
                ("unclassified", &ownership.unclassified_space),
                ("trailing_slack", &ownership.trailing_slack_space),
            ] {
                out.push_str(&format!(
                    "  {label}: sections={} bytes={}\n",
                    sections.len(),
                    sections.iter().map(|s| u128::from(s.1)).sum::<u128>()
                ));
            }
        } else {
            out.push_str("ownership (Rust): unavailable: in-place editing requires 8-byte offsets and lengths\n");
        }
        out.push_str(&format!(
            "exact_section_map_equal={}\n",
            self.rust.sections == self.c.sections
        ));
        out
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
    /// A Rust file or ownership operation failed.
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
