//! Assertions and range coverage for typed free-section reports.

/// Asserts exact section boundaries, count, and total bytes.
///
/// Accepts a borrowed [`hdf5_pure::FreeSections`] report, borrowed `(u64, u64)` address-length
/// pairs in report order, and the section address type. The type must match the sections.
///
/// # Panics
///
/// Panics if a section address or length differs, or the count or byte total differs.
///
/// # Examples
///
/// ```
/// use hdf5_pure::File;
/// use hdf5_pure::PhysicalAddress;
/// use test_util_hdf5::paged;
///
/// # let directory = tempfile::tempdir()?;
/// # let path = directory.path().join("sections.h5");
/// paged::write_metadata_tail(&path, 512, 0, 1, 18, 512);
/// let file = File::open_rw(&path)?;
/// let sections = file.reusable_free_sections()?;
/// test_util_hdf5::assert_free_sections!(
///     &sections,
///     &[(513, 511)],
///     PhysicalAddress,
/// );
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[macro_export]
macro_rules! assert_free_sections {
    ($sections:expr, $expected:expr, $address:ty $(,)?) => {{
        let sections = $sections;
        let expected = $expected;
        ::core::assert_eq!(sections.section_count(), expected.len());
        ::core::assert_eq!(
            sections.free_bytes(),
            expected.iter().map(|&(_, length)| length).sum::<u64>()
        );
        for (section, &(address, length)) in sections.sections().iter().zip(expected) {
            let _: $address = section.address();
            ::core::assert_eq!(section.address().get(), address);
            ::core::assert_eq!(section.length().get(), length);
        }
    }};
}

/// Asserts agreement between reusable-space statistics and sorted, disjoint sections.
///
/// Accepts borrowed [`hdf5_pure::ReusableSpaceStats`] and [`hdf5_pure::FreeSections`] reports
/// and the section address type, which must match the sections. Checks section ends against the
/// logical file length and
/// compares the scalar reusable-byte total with the summed section lengths.
///
/// # Panics
///
/// Panics if totals differ, sections overlap, or sections exceed the logical file length.
///
/// # Examples
///
/// ```
/// use hdf5_pure::File;
/// use hdf5_pure::PhysicalAddress;
/// use test_util_hdf5::paged;
///
/// # let directory = tempfile::tempdir()?;
/// # let path = directory.path().join("statistics.h5");
/// paged::write_metadata_tail(&path, 512, 0, 1, 18, 512);
/// let file = File::open_rw(&path)?;
/// let stats = file.reusable_space_stats()?;
/// let sections = file.reusable_free_sections()?;
/// test_util_hdf5::assert_free_sections_consistent!(
///     &stats,
///     &sections,
///     PhysicalAddress,
/// );
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[macro_export]
macro_rules! assert_free_sections_consistent {
    ($stats:expr, $sections:expr, $address:ty $(,)?) => {{
        let stats = $stats;
        let sections = $sections;
        ::core::assert_eq!(stats.reusable_free_bytes(), sections.free_bytes());
        ::core::assert_eq!(
            sections.free_bytes(),
            sections
                .sections()
                .iter()
                .map(|section| section.length().get())
                .sum::<u64>()
        );
        ::core::assert!(
            sections
                .sections()
                .windows(2)
                .all(|pair| pair[0].end_address() <= pair[1].address())
        );
        ::core::assert!(sections.sections().iter().all(|section| {
            let _: $address = section.address();
            section.end_address().get() <= stats.logical_size()
        }));
    }};
}

/// Returns the sum of section intersections with a byte range.
///
/// Accepts a borrowed [`hdf5_pure::FreeSections`] report, a borrowed [`core::ops::Range<u64>`],
/// and the section address type, which must match the sections. The range uses the same
/// coordinates as the sections.
///
/// # Examples
///
/// ```
/// use hdf5_pure::File;
/// use hdf5_pure::PhysicalAddress;
/// use test_util_hdf5::paged;
///
/// # let directory = tempfile::tempdir()?;
/// # let path = directory.path().join("coverage.h5");
/// paged::write_metadata_tail(&path, 512, 0, 1, 18, 512);
/// let file = File::open_rw(&path)?;
/// let sections = file.reusable_free_sections()?;
/// let covered = test_util_hdf5::covered_free_bytes!(
///     &sections,
///     &(500..600),
///     PhysicalAddress,
/// );
/// assert_eq!(covered, 87);
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[macro_export]
macro_rules! covered_free_bytes {
    ($sections:expr, $extent:expr, $address:ty $(,)?) => {{
        let sections = $sections;
        let extent = $extent;
        sections
            .sections()
            .iter()
            .map(|section| {
                let _: $address = section.address();
                section
                    .end_address()
                    .get()
                    .min(extent.end)
                    .saturating_sub(section.address().get().max(extent.start))
            })
            .sum::<u64>()
    }};
}
