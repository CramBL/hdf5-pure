/// The widths a file's superblock gives its offset and length fields, which
/// every structure below the superblock encodes its addresses and sizes in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Widths {
    pub offset: usize,
    pub length: usize,
}

impl Widths {
    pub const fn new(offset: usize, length: usize) -> Self {
        Self { offset, length }
    }

    /// Eight bytes each: what this workspace's writer emits, and what the
    /// reference library writes unless a creation property says otherwise.
    pub const EIGHT: Self = Self::new(8, 8);

    /// Four bytes each, the width a file written for a 32-bit address space
    /// carries.
    pub const FOUR: Self = Self::new(4, 4);
}
