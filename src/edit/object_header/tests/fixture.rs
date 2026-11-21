use crate::edit::object_header::OhRegion;
use crate::error::Error;
use crate::message_type::MessageType;

impl OhRegion {
    /// Returns a copy containing complete non-NIL records and the same prefix properties.
    ///
    /// Omits any trailing gap after the last complete record.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a record extends past the region.
    pub(in crate::edit) fn without_nil(&self) -> Result<Self, Error> {
        let mut bytes = Vec::new();
        let mut start = 0;
        while let Some((kind, _, end)) = self.next_message(start)? {
            if kind != MessageType::NIL {
                bytes.extend_from_slice(&self[start..end]);
            }
            start = end;
        }
        Ok(self.with_bytes(bytes))
    }
}
