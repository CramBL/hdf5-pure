use core::cell::Cell;

use hdf5_pure_core::__private::{FileRange, FileRegion};
use hdf5_pure_core::PhysicalAddress;
use hdf5_pure_format::__private::MetadataSource;

use super::{MAX_METADATA_BYTES, ResidualSource};
use crate::error::FormatError;

#[test]
fn residual_group_metadata_preserves_physical_bounds_and_read_budget() {
    let start = PhysicalAddress::new(u64::from(u32::MAX) + 1);
    let range = FileRange::new(start, 4).unwrap();
    let budget = Cell::new(MAX_METADATA_BYTES);
    let source = ResidualSource {
        region: FileRegion::new(range, &[1, 2, 3, 4]).unwrap(),
        remaining: &budget,
    };
    assert_eq!(source.read_metadata(start.get() + 1, 2).unwrap(), [2, 3]);
    assert_eq!(source.read_metadata_at(start.get() + 3, 1).unwrap(), [4]);
    assert_eq!(budget.get(), MAX_METADATA_BYTES - 3);
    let error = source.read_metadata(start.get() - 1, 1).unwrap_err();
    assert_eq!(
        error,
        FormatError::InsufficientRegion {
            required_start: start.get() - 1,
            required_length: 1,
            provided_start: start.get(),
            provided_length: 4
        }
    );
    budget.set(2);
    let error = source.read_metadata(start.get(), 4).unwrap_err();
    assert_eq!(
        error,
        FormatError::ResourceLimit {
            limit: u64::try_from(MAX_METADATA_BYTES).unwrap(),
            required: u64::try_from(MAX_METADATA_BYTES + 2).unwrap()
        }
    );
    assert_eq!(budget.get(), 2);
}

#[test]
fn residual_group_metadata_accepts_the_remaining_budget_before_rejecting_another_read() {
    let start = PhysicalAddress::new(64);
    let budget = Cell::new(MAX_METADATA_BYTES);
    let source = ResidualSource {
        region: FileRegion::new(FileRange::new(start, 4).unwrap(), &[1, 2, 3, 4]).unwrap(),
        remaining: &budget,
    };
    for _ in 0..MAX_METADATA_BYTES / 4 {
        assert_eq!(source.read_metadata(start.get(), 4).unwrap(), [1, 2, 3, 4]);
    }
    assert_eq!(budget.get(), 0);
    assert_eq!(
        source.read_metadata(start.get(), 1).unwrap_err(),
        FormatError::ResourceLimit {
            limit: u64::try_from(MAX_METADATA_BYTES).unwrap(),
            required: u64::try_from(MAX_METADATA_BYTES + 1).unwrap(),
        }
    );
    assert_eq!(budget.get(), 0);
}
