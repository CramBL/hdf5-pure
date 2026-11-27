use super::{ExtensibleArrayPlan, SlotOccupancy};
use crate::extensible_array::ExtensibleArrayCreationParameters;
use crate::width::FormatWidths;

pub(super) fn plan(creation: &[u8], slots: &[u64], count: u64, width: u8) -> ExtensibleArrayPlan {
    ExtensibleArrayPlan::new(
        ExtensibleArrayCreationParameters::parse(creation).unwrap(),
        SlotOccupancy::Slots(slots),
        count,
        8,
        FormatWidths::from_sizes(width, 8).unwrap(),
        false,
    )
    .unwrap()
}
