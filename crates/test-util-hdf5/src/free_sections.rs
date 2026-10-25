//! Assertions and range coverage for typed free-section reports.

/// Asserts exact section boundaries, coordinates, count, and total bytes.
///
/// Accepts a borrowed [`hdf5_pure::FreeSections`] report, borrowed `(u64, u64)` address-length
/// pairs in report order, and the expected [`hdf5_pure::FreeSectionCoordinates`].
///
/// # Panics
///
/// Panics if a section address, length, or coordinate differs, or the count or byte total differs.
///
/// # Examples
///
/// ```
/// use hdf5_pure::File;
/// use hdf5_pure::FreeSectionCoordinates;
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
///     FreeSectionCoordinates::Absolute,
/// );
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[macro_export]
macro_rules! assert_free_sections {
    ($sections:expr, $expected:expr, $coordinates:expr $(,)?) => {{
        let sections = $sections;
        let expected = $expected;
        let coordinates = $coordinates;
        ::core::assert_eq!(sections.section_count(), expected.len());
        ::core::assert_eq!(
            sections.free_bytes(),
            expected.iter().map(|&(_, length)| length).sum::<u64>()
        );
        for (section, &(address, length)) in sections.sections().iter().zip(expected) {
            ::core::assert_eq!(section.coordinates(), coordinates);
            ::core::assert_eq!(section.address(), address);
            ::core::assert_eq!(section.length().get(), length);
        }
    }};
}

/// Asserts agreement between reusable-space statistics and sorted, disjoint sections.
///
/// Accepts borrowed [`hdf5_pure::ReusableSpaceStats`] and [`hdf5_pure::FreeSections`] reports
/// and the expected [`hdf5_pure::FreeSectionCoordinates`]. Checks section ends against the logical
/// file length and compares the scalar reusable-byte total with the summed section lengths.
///
/// # Panics
///
/// Panics if totals differ, sections overlap or exceed the logical file length, or coordinates
/// differ from the expected value.
///
/// # Examples
///
/// ```
/// use hdf5_pure::File;
/// use hdf5_pure::FreeSectionCoordinates;
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
///     FreeSectionCoordinates::Absolute,
/// );
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[macro_export]
macro_rules! assert_free_sections_consistent {
    ($stats:expr, $sections:expr, $coordinates:expr $(,)?) => {{
        let stats = $stats;
        let sections = $sections;
        let coordinates = $coordinates;
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
            section.coordinates() == coordinates && section.end_address() <= stats.logical_size()
        }));
    }};
}

/// Returns the sum of section intersections with a byte range.
///
/// Accepts a borrowed [`hdf5_pure::FreeSections`] report, a borrowed [`core::ops::Range<u64>`],
/// and the expected [`hdf5_pure::FreeSectionCoordinates`]. The range uses the same coordinates as
/// the sections.
///
/// # Panics
///
/// Panics if any section's coordinates differ from the expected value.
///
/// # Examples
///
/// ```
/// use hdf5_pure::File;
/// use hdf5_pure::FreeSectionCoordinates;
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
///     FreeSectionCoordinates::Absolute,
/// );
/// assert_eq!(covered, 87);
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
#[macro_export]
macro_rules! covered_free_bytes {
    ($sections:expr, $extent:expr, $coordinates:expr $(,)?) => {{
        let sections = $sections;
        let extent = $extent;
        let coordinates = $coordinates;
        sections
            .sections()
            .iter()
            .map(|section| {
                ::core::assert_eq!(section.coordinates(), coordinates);
                section
                    .end_address()
                    .min(extent.end)
                    .saturating_sub(section.address().max(extent.start))
            })
            .sum::<u64>()
    }};
}
