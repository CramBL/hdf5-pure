use crate::bytes;

/// A file image under construction, into which a test places one structure at
/// a time.
///
/// The offsets a structure's fields carry are file offsets, so a fixture made
/// of more than one structure has to decide where each lands before it can
/// write the pointers between them. Placing a block at a chosen offset grows
/// the image to reach it, and appending one reports where it landed.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Image(Vec<u8>);

impl Image {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// An image that already holds `bytes`, for a fixture built around one
    /// structure that has to sit at offset zero.
    pub fn starting_with(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }

    /// Writes `block` at `at`, zero-filling whatever gap precedes it.
    #[track_caller]
    pub fn place(&mut self, at: usize, block: &[u8]) {
        self.0.resize(self.0.len().max(at + block.len()), 0);
        bytes::set_slice_at(&mut self.0, at, block);
    }

    /// Writes `block` at the end of the image and returns its offset.
    pub fn append(&mut self, block: &[u8]) -> usize {
        let at = self.0.len();
        self.0.extend_from_slice(block);
        at
    }

    /// Writes `block` at the next offset that is a multiple of `alignment` and
    /// returns it, for the structures a file keeps aligned.
    #[track_caller]
    pub fn append_aligned(&mut self, block: &[u8], alignment: usize) -> usize {
        assert!(alignment > 0, "an alignment is at least one byte");
        let at = self.0.len().next_multiple_of(alignment);
        self.place(at, block);
        at
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub fn build(self) -> Vec<u8> {
        self.0
    }
}
