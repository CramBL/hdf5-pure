mod encode;
mod placement;
mod prepared;
mod serialize;

pub use encode::{
    FreeSpaceManagerEncoding, FreeSpaceManagerHeaderEncoding, FreeSpaceManagerSectionListEncoding,
};
pub use prepared::PreparedFreeSpaceManager;
pub use serialize::{serialize_free_space_manager, serialize_free_space_manager_with_allocation};
