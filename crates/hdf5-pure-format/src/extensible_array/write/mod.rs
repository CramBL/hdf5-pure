//! Complete Extensible Array block geometry and checked stored placements.

mod block;
mod encode;
mod header;
mod initialization;
mod occupancy;
mod placement;
mod plan;

#[cfg(test)]
mod test_support;

pub use block::ExtensibleArrayBlockEncoding;
pub use block::ExtensibleArrayBlockWriter;
pub use block::ExtensibleArrayPageEncoding;
pub use encode::ExtensibleArrayEmission;
pub use encode::ExtensibleArrayEncoding;
pub use header::ExtensibleArrayHeaderWithoutIndexBlock;
pub use initialization::{ExtensibleArrayInitialization, build_extensible_array_at};
pub use occupancy::{
    ExtensibleArrayLayout, SlotOccupancy, extensible_array_layout, extensible_array_len,
};
pub use placement::ExtensibleArrayPlacement;
pub use plan::{ExtensibleArrayBlock, ExtensibleArrayBlockKind, ExtensibleArrayPlan};
