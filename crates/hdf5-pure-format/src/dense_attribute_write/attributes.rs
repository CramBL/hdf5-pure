use alloc::vec::Vec;

use crate::attribute_message::AttributeMessage;
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::width::FormatWidths;

use super::fixup::FixupRange;

/// Creation indexes and the counter recorded for a dense attribute set.
///
/// A tracked set includes one index per attribute in caller order. Gaps and the next-index
/// counter are retained independently of the number of live attributes. An attribute's heap
/// ID is the same in the name index and the optional creation-order index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum DenseAttrCreationOrder {
    /// Untracked creation order, with positions recorded in name-index records.
    #[default]
    Untracked,
    /// Tracked, with one creation index per attribute in the caller's order.
    Tracked {
        /// Creation index of each attribute, parallel to the attribute slice.
        indices: Vec<u16>,
        /// The counter for the next creation index, independent of `indices`.
        max: u16,
        /// Whether to emit the creation-order B-tree beside the name index.
        indexed: bool,
    },
}

impl DenseAttrCreationOrder {
    /// Returns the tracked creation index at `i`.
    ///
    /// Returns `None` for untracked order or an index outside the set.
    pub fn index_at(&self, i: usize) -> Option<u16> {
        match self {
            Self::Untracked => None,
            Self::Tracked { indices, .. } => indices.get(i).copied(),
        }
    }
}

/// Encoded attributes with unique names and creation indexes.
///
/// Tracked indexes are below the stored counter. Gaps and the counter are preserved
/// independently of the live attributes.
#[derive(Debug)]
pub(super) struct EncodedDenseAttrs {
    widths: FormatWidths,
    attributes: Vec<EncodedDenseAttribute>,
    indexes: AttributeIndexes,
}

/// The attribute positions in name and creation order, with the tracked creation counter.
#[derive(Debug)]
pub(super) struct AttributeIndexes {
    /// `(name hash, attribute index)` pairs sorted by hash and then by name.
    order: Vec<(u32, u32)>,
    /// Attribute indexes sorted by creation index for the optional creation-order B-tree.
    corder_order: Vec<u32>,
    next_index: Option<u16>,
    indexed: bool,
}

impl AttributeIndexes {
    pub(super) fn order(&self) -> &[(u32, u32)] {
        &self.order
    }

    pub(super) fn corder_order(&self) -> &[u32] {
        &self.corder_order
    }

    pub(super) fn next_index(&self) -> Option<u16> {
        self.next_index
    }

    pub(super) fn indexed(&self) -> bool {
        self.indexed
    }
}

impl EncodedDenseAttrs {
    /// Encodes version 3 attribute bodies and pairs each with its creation index.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the counts of attributes and tracked indexes differ.
    /// Returns the metadata errors documented by [`dense_attrs_plan`](super::dense_attrs_plan)
    /// for repeated names or indexes and inconsistent counters. Propagates errors from attribute
    /// encoding and position conversion.
    pub(super) fn new(
        widths: FormatWidths,
        attrs: &[AttributeMessage],
        creation: DenseAttrCreationOrder,
    ) -> Result<Self, FormatError> {
        let (indices, next_index, indexed) = match creation {
            DenseAttrCreationOrder::Untracked => (None, None, false),
            DenseAttrCreationOrder::Tracked {
                indices,
                max,
                indexed,
            } => {
                if indices.len() != attrs.len() {
                    return Err(FormatError::Internal(
                        "a tracked attribute set needs one creation index per attribute".into(),
                    ));
                }
                (Some(indices), Some(max), indexed)
            }
        };
        let mut attributes = Vec::with_capacity(attrs.len());
        let mut order = Vec::with_capacity(attrs.len());
        for (i, attribute) in attrs.iter().enumerate() {
            let position = i.narrow::<u32>()?;
            let creation_index = match &indices {
                Some(indices) => {
                    let index = indices[i];
                    if let Some(next) = next_index {
                        if index >= next {
                            return Err(FormatError::InvalidAttributeCreationCounter {
                                index,
                                next,
                            });
                        }
                    }
                    u32::from(index)
                }
                None => position,
            };
            attributes.push(EncodedDenseAttribute {
                bytes: attribute.prepare_v3(widths)?.to_bytes(),
                creation_index,
            });
            order.push((
                crate::checksum::jenkins_lookup3(attribute.name.as_bytes()),
                position,
            ));
        }
        order.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| attrs[a.1 as usize].name.cmp(&attrs[b.1 as usize].name))
        });
        for pair in order.windows(2) {
            let left = &attrs[pair[0].1 as usize].name;
            let right = &attrs[pair[1].1 as usize].name;
            if left == right {
                return Err(FormatError::DuplicateAttributeName(left.clone()));
            }
        }
        let mut corder_order = if next_index.is_some() {
            (0..attrs.len().narrow::<u32>()?).collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        corder_order.sort_unstable_by_key(|&i| attributes[i as usize].creation_index());
        for pair in corder_order.windows(2) {
            let left = attributes[pair[0] as usize].creation_index;
            let right = attributes[pair[1] as usize].creation_index;
            if left == right {
                return Err(FormatError::DuplicateAttributeCreationIndex(left));
            }
        }
        Ok(Self {
            widths,
            attributes,
            indexes: AttributeIndexes {
                order,
                corder_order: if indexed { corder_order } else { Vec::new() },
                next_index,
                indexed,
            },
        })
    }

    pub(super) fn into_parts(self) -> (FormatWidths, Vec<EncodedDenseAttribute>, AttributeIndexes) {
        (self.widths, self.attributes, self.indexes)
    }
}

/// A version 3 attribute message paired with its creation index.
///
/// For untracked attributes, the creation index is the position in the caller's set.
#[derive(Clone, Debug)]
pub(super) struct EncodedDenseAttribute {
    bytes: Vec<u8>,
    creation_index: u32,
}

impl EncodedDenseAttribute {
    pub(super) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Borrows the bytes reserved for a deferred update in one encoded attribute.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::Internal`] if the attribute position or byte range is out of bounds.
    pub(super) fn fixup_bytes_mut<'a>(
        attributes: &'a mut [Self],
        field: &FixupRange,
    ) -> Result<&'a mut [u8], FormatError> {
        attributes
            .get_mut(field.attribute())
            .and_then(|attribute| attribute.bytes.get_mut(field.bytes()))
            .ok_or_else(|| {
                FormatError::Internal("a prepared fixup lies outside its encoded attribute".into())
            })
    }

    pub(super) fn creation_index(&self) -> u32 {
        self.creation_index
    }
}

#[cfg(test)]
#[path = "tests/attributes.rs"]
mod tests;
