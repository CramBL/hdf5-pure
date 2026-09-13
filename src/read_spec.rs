//! The per-dataset properties every raw read is performed against.
//!
//! Reading a dataset's element bytes takes five facts about the dataset — where
//! its storage is, what shape it has, what an element is, what filters its bytes
//! passed through, and what unallocated storage reads as — and the call chain
//! that does it is four layers deep: the reader picks a backend, `data_read`
//! dispatches on the layout, `chunked_read` walks the chunk index, and the
//! assembly kernel places what it decoded. Every layer needs all five, so before
//! this type they were threaded through as five positional parameters and the
//! signatures reached twelve arguments.
//!
//! The cost of that shape was not the width; it was that adding a property meant
//! editing roughly thirty call sites, several of them mechanically across test
//! fixtures, where a default inserted in place of a real value compiles and
//! passes (issue #294). A property added to [`RawReadSpec`] instead reaches every
//! layer at once, and the sites that must decide what it is are the few that
//! build the spec.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::convert::TryToUsize;
use crate::data_layout::DataLayout;
use crate::dataspace::Dataspace;
use crate::datatype::Datatype;
use crate::error::FormatError;
use crate::fill_value::FillPattern;
use crate::filter_pipeline::FilterPipeline;

/// What a raw read needs to know about the dataset it is reading.
///
/// These are properties of the *dataset*. The file-level address widths
/// (`offset_size`, `length_size`) travel beside a spec rather than in it: they
/// are the same for every dataset in a file, and the rest of the crate already
/// passes them as a pair.
///
/// `Copy`, so a spec is handed down the call chain the way the five borrows it
/// replaced were, with no clone and no lifetime beyond the borrows themselves.
#[derive(Clone, Copy)]
pub struct RawReadSpec<'a> {
    /// Where the dataset's bytes live: compact, contiguous, or chunked.
    pub layout: &'a DataLayout,
    /// The dataset's shape, which fixes how many elements the read produces.
    pub dataspace: &'a Dataspace,
    /// The stored element type, which fixes each element's width.
    pub datatype: &'a Datatype,
    /// The filter pipeline stored bytes passed through, if any.
    pub pipeline: Option<&'a FilterPipeline>,
    /// What storage that was never allocated reads as.
    pub fill: FillPattern<'a>,
}

impl<'a> RawReadSpec<'a> {
    /// The byte length this dataset's elements occupy, checked against what its
    /// layout says it stores.
    ///
    /// A compact or contiguous layout names its own size, and a disagreement
    /// with the dataspace is a corrupt or crafted file rather than a short read.
    /// The check lives here because two kinds of read have to make it and make
    /// it the same way: a whole read, and a read that takes the dataset a row
    /// window at a time. A check only one of them made would fire or not
    /// depending on the dataset's *size* — the windowed readers read a small
    /// dataset whole — which is worse than no check at all (issue #289).
    ///
    /// A zero-element dataset is exempt, as it is in the readers: it has no
    /// bytes to disagree about, and its layout may say anything.
    ///
    /// Chunked storage names no total size, and a chunk the file does not hold
    /// reads as the fill value rather than as an error, so there is nothing here
    /// to check for it.
    pub fn stored_byte_len(&self) -> Result<usize, FormatError> {
        let num_elements = self.dataspace.num_elements().to_usize()?;
        let elem_size = self.datatype.type_size() as usize;
        let expected = num_elements
            .checked_mul(elem_size)
            .ok_or(FormatError::OffsetOverflow {
                offset: num_elements as u64,
                length: elem_size as u64,
            })?;
        if num_elements == 0 {
            return Ok(expected);
        }
        let actual = match self.layout {
            DataLayout::Compact { data } => data.len(),
            // No address means storage that was never allocated, which reads as
            // the fill value and has no size to disagree with.
            DataLayout::Contiguous {
                address: Some(_),
                size,
            } => (*size).to_usize()?,
            _ => return Ok(expected),
        };
        if actual != expected {
            return Err(FormatError::DataSizeMismatch { expected, actual });
        }
        Ok(expected)
    }

    /// Returns the buffer a dataset whose storage was never allocated reads as.
    ///
    /// A chunked dataset's index is allocated lazily, since the reference C
    /// library leaves the layout message's address undefined until the first
    /// chunk is written, so a dataset that has been created but never written
    /// refers to no index at all. That is neither an error nor an empty read: it
    /// reads as one `fill` element per element of its dataspace, which for a
    /// zero-element dataset is the empty buffer and for any other is a whole
    /// materialized dataset.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::UnreadableFillValue`] if the dataset's Fill Value
    /// message could not be parsed, leaving those elements no value to read as,
    /// [`FormatError::ZeroSizedDatatype`] if the datatype occupies zero bytes
    /// per element, and [`FormatError::ValueTooLargeForPlatform`] or
    /// [`FormatError::OffsetOverflow`] if the materialized dataset would be
    /// larger than this target can address.
    pub(crate) fn unallocated_buffer(&self) -> Result<Vec<u8>, FormatError> {
        let elem_size = self.datatype.element_size_usize()?;
        let total = self
            .dataspace
            .num_elements()
            .to_usize()?
            .checked_mul(elem_size.get())
            .ok_or(FormatError::OffsetOverflow {
                offset: self.dataspace.num_elements(),
                length: elem_size.get() as u64,
            })?;
        self.fill.buffer(total)
    }

    /// A spec for an unfiltered dataset whose unallocated storage reads as
    /// zeros — the shape a test fixture wants, and never the shape a real read
    /// wants, which is why the live paths build the struct literally.
    #[cfg(test)]
    pub fn plain(layout: &'a DataLayout, dataspace: &'a Dataspace, datatype: &'a Datatype) -> Self {
        Self {
            layout,
            dataspace,
            datatype,
            pipeline: None,
            fill: FillPattern::ZERO,
        }
    }
}
