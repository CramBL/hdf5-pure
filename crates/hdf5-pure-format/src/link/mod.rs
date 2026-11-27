//! Version 1 Link messages, group link storage, and dense link indexes.

pub(crate) mod creation_order;
pub(crate) mod dense;
pub(crate) mod encode;
mod error;
pub(crate) mod info;
mod name;
mod projection;
mod source;
pub(crate) mod storage;
mod view;
mod wire;

pub use encode::{LinkOutputTooSmall, PreparedLink};
pub use error::LinkWireError;
pub use name::{LinkName, LinkTargetV1, UserLinkKind, UserLinkValue};
pub use projection::{
    LinkMessage, LinkProjectionError, LinkTarget, SymbolicLinkTarget, link_is_named,
};
pub(crate) use source::StoredLink;
pub use view::{LinkMessageView, LinkTargetView};
pub use wire::LinkMessageV1;
