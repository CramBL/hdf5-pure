//! The continuation blocks of a version 2 object header that a parse has found, in the order it
//! reads them.

use alloc::vec::Vec;

use crate::error::FormatError;
use crate::object_header::v2::ObjectHeaderContinuation;

/// The continuation messages a parse has found, in the order the parse reads their blocks.
///
/// [`next_unvisited`](Self::next_unvisited) returns the messages in the order
/// [`discover`](Self::discover) receives them, so the blocks a continuation block refers to come
/// after every block found before them.
///
/// The C library reads the blocks in the same order. `H5O__add_cont_msg` appends each continuation
/// message the library decodes to one list (`H5Ocache.c`), and `H5O_protect` reads the blocks from
/// the front of that list as the list grows by the messages of each block it reads (`H5Oint.c`,
/// HDF5 2.2.0).
pub(super) struct ContinuationWorklist {
    discovered: Vec<ObjectHeaderContinuation>,
    /// The number of messages `next_unvisited` has returned, which is the index of the next one in
    /// `discovered`.
    visited: usize,
}

impl ContinuationWorklist {
    pub(super) fn new() -> Self {
        Self {
            discovered: Vec::new(),
            visited: 0,
        }
    }

    /// Adds `continuation` after the continuation messages found before it.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::NestingDepthExceeded`] if the parse has found [`MAX_CONTINUATIONS`]
    /// continuation messages already.
    pub(super) fn discover(
        &mut self,
        continuation: ObjectHeaderContinuation,
    ) -> Result<(), FormatError> {
        if self.discovered.len() >= MAX_CONTINUATIONS {
            return Err(FormatError::NestingDepthExceeded);
        }
        self.discovered.push(continuation);
        Ok(())
    }

    /// Returns the next continuation message in the order the parse found them, or `None` once
    /// every one found is returned.
    pub(super) fn next_unvisited(&mut self) -> Option<ObjectHeaderContinuation> {
        let next = self.discovered.get(self.visited).copied()?;
        self.visited += 1;
        Some(next)
    }
}

/// The most continuation messages a parse accepts in one version 2 object header, across chunk 0
/// and its continuation blocks.
///
/// [`ContinuationWorklist::discover`] rejects the next one, which ends the parse of a header whose
/// blocks refer to each other in a cycle.
const MAX_CONTINUATIONS: usize = 256;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::StoredAddress;
    use crate::convert::Narrow;

    #[test]
    fn a_continuation_found_while_visiting_comes_after_those_found_before_it() {
        let [a, b, a1] = [0x100, 0x200, 0x300].map(continuation);
        let mut worklist = ContinuationWorklist::new();
        worklist.discover(a).unwrap();
        worklist.discover(b).unwrap();

        assert_eq!(worklist.next_unvisited(), Some(a));
        worklist.discover(a1).unwrap();
        assert_eq!(worklist.next_unvisited(), Some(b));
        assert_eq!(worklist.next_unvisited(), Some(a1));
        assert_eq!(worklist.next_unvisited(), None);
    }

    #[test]
    fn a_continuation_past_the_limit_is_rejected() {
        let mut worklist = ContinuationWorklist::new();
        (0..MAX_CONTINUATIONS.to_u64())
            .map(continuation)
            .try_for_each(|found| worklist.discover(found))
            .unwrap();

        assert_eq!(
            worklist.discover(continuation(0)),
            Err(FormatError::NestingDepthExceeded)
        );
    }

    fn continuation(address: u64) -> ObjectHeaderContinuation {
        ObjectHeaderContinuation {
            address: StoredAddress::new(address),
            length: 16,
        }
    }
}
