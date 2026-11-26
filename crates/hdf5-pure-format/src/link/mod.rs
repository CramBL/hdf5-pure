//! Version 1 Link messages, borrowed fields, and text projections.

pub(crate) mod encode;
mod error;
mod name;
mod projection;
mod source;
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
