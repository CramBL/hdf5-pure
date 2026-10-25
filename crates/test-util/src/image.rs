use crate::bytes;
use crate::link_message::HardLink;
use crate::object_header::Message;
use crate::object_header::MessageType;
use crate::object_header::v2::Header;
use crate::superblock::v2::Superblock;
use crate::widths::Widths;

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

    /// Finishes the image as a file whose root group links a dataset as `link_name`, and returns
    /// its bytes.
    ///
    /// The object header of the dataset holds `messages`, and the object header of the root group
    /// holds one Link message to it. Both are version 2 object headers appended to the image, and
    /// a version 2 superblock with 8-byte offsets and lengths is written at offset 0.
    ///
    /// # Panics
    ///
    /// Panics if the image does not start with as many zero bytes as the superblock is long, the
    /// space a fixture reserves with [`Image::starting_with`].
    #[track_caller]
    pub fn build_with_dataset(
        mut self,
        link_name: &str,
        messages: impl IntoIterator<Item = Message>,
    ) -> Vec<u8> {
        let dataset = self.append(&Header::new().messages(messages).build());
        let root = self.append(
            &Header::new()
                .message(Message::new(
                    MessageType::LINK,
                    &HardLink::new(link_name, dataset as u64).build(Widths::EIGHT),
                ))
                .build(),
        );
        let superblock = Superblock::new(Widths::EIGHT)
            .root_header_address(root as u64)
            .eof_address(self.0.len() as u64)
            .build();
        assert!(
            self.0
                .get(..superblock.len())
                .is_some_and(|reserved| reserved.iter().all(|&byte| byte == 0)),
            "the image reserves {} zero bytes at offset 0 for the superblock",
            superblock.len()
        );
        self.place(0, &superblock);
        self.0
    }
}
