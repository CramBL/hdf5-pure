//! Test counters for successful offset- and length-field writes.

use core::cell::Cell;

use crate::width::{LengthWidth, OffsetWidth};

/// Recorded field writes and their combined encoded size.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Work {
    /// Offset-width fields written.
    offsets: u64,
    /// Length-width fields written.
    lengths: u64,
    /// Bytes written in the recorded fields.
    bytes: u64,
}

std::thread_local! {
    static WORK: Cell<Work> = const { Cell::new(Work { offsets: 0, lengths: 0, bytes: 0 }) };
}

/// Records one offset-width field written on the current thread.
pub(crate) fn offset(width: OffsetWidth) {
    WORK.with(|observed| {
        let mut work = observed.get();
        work.offsets += 1;
        work.bytes += u64::from(width.get());
        observed.set(work);
    });
}

/// Records one length-width field written on the current thread.
pub(crate) fn length(width: LengthWidth) {
    WORK.with(|observed| {
        let mut work = observed.get();
        work.lengths += 1;
        work.bytes += u64::from(width.get());
        observed.set(work);
    });
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::Cell;

    use rstest::rstest;
    use test_util::dataspace;

    use super::{WORK, Work};
    use crate::address::StoredAddress;
    use crate::address::StoredAddressExt;
    use crate::chunk_record::{ChunkRecord, IndexSlots};
    use crate::dataspace::{Dataspace, DataspaceType, MaxExtent};
    use crate::fixed_array::FixedArrayPlan;
    use crate::free_space_manager::{self, FreeSection};
    use crate::width::{FormatWidths, LengthWidth, OffsetWidth};

    #[rstest]
    #[case::two_eight(OffsetWidth::Two, LengthWidth::Eight)]
    #[case::eight_two(OffsetWidth::Eight, LengthWidth::Two)]
    #[case::default(OffsetWidth::Eight, LengthWidth::Eight)]
    fn complete_encodings_do_constant_work_per_file_width_field(
        #[case] offsets: OffsetWidth,
        #[case] lengths: LengthWidth,
        #[values(1, 8, 64)] entries: u64,
        #[values(false, true)] mixed: bool,
    ) {
        let widths = FormatWidths { offsets, lengths };
        WORK.with(|work| work.set(Work::default()));
        let bytes = encode(widths, entries, mixed);
        let observed = WORK.with(Cell::get);
        // The writer encodes entries + 2 offsets per array and manager, plus twelve lengths.
        assert_eq!(
            observed,
            Work {
                offsets: 2 * entries + 4,
                lengths: 12,
                bytes: (2 * entries + 4) * u64::from(offsets.get()) + 12 * u64::from(lengths.get()),
            }
        );
        assert!(u64::try_from(bytes).unwrap() > observed.bytes);
        WORK.with(|work| work.set(Work::default()));
        assert_eq!(encode(widths, entries, mixed), bytes);
        assert_eq!(encode(widths, entries, mixed), bytes);
        let control = WORK.with(Cell::get);
        assert_eq!(control.offsets, 2 * observed.offsets);
        assert_eq!(control.lengths, 2 * observed.lengths);
        assert_eq!(control.bytes, 2 * observed.bytes);
        assert!(control.bytes > observed.bytes);
    }

    fn encode(widths: FormatWidths, entries: u64, mixed: bool) -> usize {
        let dataspace = Dataspace {
            space_type: DataspaceType::Simple,
            rank: 2,
            dimensions: vec![3, 5],
            max_dimensions: Some(vec![MaxExtent::Fixed(7), MaxExtent::Fixed(9)]),
        }
        .serialize(widths.lengths.get())
        .unwrap();
        let occupied = (0..entries).collect::<Vec<_>>();
        let records = occupied
            .iter()
            .map(|&slot| ChunkRecord {
                address: StoredAddress::new(1024 + 4 * slot),
                stored_size: 4,
                filter_mask: 0,
            })
            .collect::<Vec<_>>();
        let slots = IndexSlots::new(&records, &occupied, entries).unwrap();
        let plan = FixedArrayPlan::new(entries, 4, widths, false).unwrap();
        let header_length = plan.block_lengths().next().unwrap();
        let blocks = plan
            .place(
                StoredAddress::new(4096),
                StoredAddress::new(4096 + header_length),
            )
            .unwrap()
            .serialize_blocks(&slots)
            .unwrap();
        let sections = occupied
            .iter()
            .map(|&slot| FreeSection {
                addr: StoredAddress::new(8192 + 32 * slot),
                size: 16 + if mixed { 8 * (slot % 2) } else { 0 },
            })
            .collect::<Vec<_>>();
        let (header, list) = free_space_manager::serialize_free_space_manager(
            widths,
            &sections,
            StoredAddress::new(256),
            StoredAddress::new(512),
            free_space_manager::SECTION_CLASS_SIMPLE,
        )
        .unwrap();
        assert_eq!(
            dataspace.len(),
            dataspace::V2_HEADER_LEN + 4 * usize::from(widths.lengths.get())
        );
        assert_eq!(
            blocks.iter().map(|(_, bytes)| bytes.len()).sum::<usize>(),
            usize::try_from(plan.total_len()).unwrap()
        );
        dataspace.len()
            + blocks.iter().map(|(_, bytes)| bytes.len()).sum::<usize>()
            + header.len()
            + list.len()
    }
}
