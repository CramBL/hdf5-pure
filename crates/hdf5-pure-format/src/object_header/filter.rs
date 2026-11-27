use crate::message_type::MessageType;

/// Selects the message bodies to copy into the parsed object header.
///
/// The parser follows continuation records regardless of which messages the filter keeps.
pub enum MessageFilter<'f> {
    /// Keeps every message offered to the filter.
    All,
    /// Keeps a message when the predicate returns true for its type and body.
    Only(&'f mut dyn FnMut(MessageType, &[u8]) -> bool),
}

impl MessageFilter<'_> {
    /// Returns whether to retain this message.
    pub(super) fn keeps(&mut self, msg_type: MessageType, data: &[u8]) -> bool {
        match self {
            MessageFilter::All => true,
            MessageFilter::Only(f) => f(msg_type, data),
        }
    }

    /// Returns whether the parser can reserve space for every message in a chunk.
    pub(super) fn keeps_all(&self) -> bool {
        matches!(self, MessageFilter::All)
    }
}
