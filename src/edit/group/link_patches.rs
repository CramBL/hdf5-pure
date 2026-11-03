use core::cmp::Ordering;
use core::ops::Range;
use std::vec;

use crate::address::StoredAddress;
use crate::edit::object_header::OhRegion;
use crate::error::Error;
use crate::error::FormatError;
use crate::file_writer::OFFSET_WIDTH;
use crate::link_message::LinkMessage;
use crate::message_type::MessageType;

/// Hard-link address fields prepared for one borrowed working header.
///
/// Requests are applied in input order, including repeated names. Mutation is limited to
/// address fields and appended bytes, preserving the prepared field locations. Preparation
/// failures are reported by [`Self::patch_next`].
pub(in crate::edit) struct LinkPatches<'a> {
    region: &'a mut OhRegion,
    fields: PatchFields,
    framing_error: Option<Error>,
    #[cfg(test)]
    visited: usize,
}

impl<'a> LinkPatches<'a> {
    /// Resolves requested names in one record scan without copying names or symbolic targets.
    ///
    /// Defers patch failures to [`Self::patch_next`]. Empty requests skip the scan, and a
    /// single request uses inline storage. Names are compared with replacement characters for
    /// invalid UTF-8, as in [`LinkMessage::parse`].
    pub(in crate::edit) fn new<'name>(
        region: &'a mut OhRegion,
        names: impl IntoIterator<Item = &'name str>,
    ) -> Self {
        let mut names = names.into_iter();
        let Some(first) = names.next() else {
            return Self {
                region,
                fields: PatchFields::Empty,
                framing_error: None,
                #[cfg(test)]
                visited: 0,
            };
        };
        let Some(second) = names.next() else {
            let mut field = PatchField::Missing;
            let (framing_error, _visited) = Self::prepare(
                region,
                &[(LossyName(first.as_bytes()), 0)],
                core::slice::from_mut(&mut field),
            );
            return Self {
                region,
                fields: PatchFields::One(Some(field)),
                framing_error,
                #[cfg(test)]
                visited: _visited,
            };
        };
        let mut requests: Vec<_> = core::iter::once(first)
            .chain(core::iter::once(second))
            .chain(names)
            .enumerate()
            .map(|(index, name)| (LossyName(name.as_bytes()), index))
            .collect();
        requests.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        let mut fields = vec![PatchField::Missing; requests.len()];
        let (framing_error, _visited) = Self::prepare(region, &requests, &mut fields);
        Self {
            region,
            fields: PatchFields::Many(fields.into_iter()),
            framing_error,
            #[cfg(test)]
            visited: _visited,
        }
    }

    /// Resolves sorted requests to fields in their original order and counts visited records.
    ///
    /// Returns the first record-framing error alongside the visit count. Unparseable Link
    /// bodies are skipped. Each request resolves to the first matching parsed link.
    fn prepare(
        region: &OhRegion,
        requests: &[(LossyName<'_>, usize)],
        fields: &mut [PatchField],
    ) -> (Option<Error>, usize) {
        let mut remaining = fields.len();
        let mut framing_error = None;
        let mut start = 0;
        let mut visited = 0;
        while remaining != 0 {
            let (kind, body, end) = match region.next_message(start) {
                Ok(Some(record)) => record,
                Ok(None) => break,
                Err(error) => {
                    framing_error = Some(error);
                    break;
                }
            };
            visited += 1;
            if kind == MessageType::LINK
                && let Ok(name) = LinkMessage::parsed_name_bytes(&region[body..end], OFFSET_WIDTH)
            {
                let name = LossyName(name);
                let first = requests.partition_point(|request| request.0 < name);
                let last = requests.partition_point(|request| request.0 <= name);
                if first != last && matches!(fields[requests[first].1], PatchField::Missing) {
                    let field = match LinkMessage::hard_link_address_range(
                        &region[body..end],
                        OFFSET_WIDTH,
                    ) {
                        Ok(Some(address)) if body + address.end == end => {
                            PatchField::Address(body + address.start..end)
                        }
                        Ok(Some(_)) => PatchField::Trailing,
                        Ok(None) => PatchField::Symbolic,
                        Err(_) => {
                            start = end;
                            continue;
                        }
                    };
                    for &(_, index) in &requests[first..last] {
                        if matches!(fields[index], PatchField::Missing) {
                            fields[index] = field.clone();
                            remaining -= 1;
                        }
                    }
                }
            }
            start = end;
        }
        (framing_error, visited)
    }

    /// Writes `address` to the next requested link, consuming the request even on failure.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EditUnsupported`] if a named link is absent or symbolic, its
    /// address is followed by trailing body bytes, or record framing prevents finding it.
    /// Returns [`FormatError::Internal`] if no request remains or a prepared field is outside
    /// the borrowed region. Propagates address-encoding errors.
    pub(in crate::edit) fn patch_next(&mut self, address: StoredAddress) -> Result<(), Error> {
        let Self {
            region,
            fields,
            framing_error,
            #[cfg(test)]
                visited: _,
        } = self;
        match fields.next() {
            Some(PatchField::Address(range)) => {
                hdf5_pure_format::__private::write_stored_address_at(
                    region.bytes_mut().get_mut(range).ok_or_else(|| {
                        FormatError::Internal(
                            "a prepared link address is outside its region".into(),
                        )
                    })?,
                    0,
                    address,
                    OFFSET_WIDTH,
                )?;
                Ok(())
            }
            Some(PatchField::Symbolic) => Err(Error::EditUnsupported(
                "a group on the edited path is reached by a soft/external link",
            )),
            Some(PatchField::Trailing) => Err(Error::EditUnsupported(
                "a link message holds bytes past its hard link address",
            )),
            Some(PatchField::Missing) => Err(framing_error.take().unwrap_or(
                Error::EditUnsupported("expected child link not found in parent group"),
            )),
            None => Err(FormatError::Internal(
                "a group has no remaining prepared link patch".into(),
            )
            .into()),
        }
    }

    /// Appends bytes without changing the prepared fields or adding patch requests.
    pub(in crate::edit) fn push_bytes(&mut self, bytes: &[u8]) {
        self.region.push_bytes(bytes);
    }
}

/// Pending fields in request order, with inline storage for at most one request.
enum PatchFields {
    Empty,
    Many(vec::IntoIter<PatchField>),
    One(Option<PatchField>),
}

impl PatchFields {
    fn next(&mut self) -> Option<PatchField> {
        match self {
            Self::Empty => None,
            Self::Many(fields) => fields.next(),
            Self::One(field) => field.take(),
        }
    }
}

/// A checked address range or the reason a requested link cannot be patched.
#[derive(Clone)]
enum PatchField {
    Address(Range<usize>),
    Missing,
    Symbolic,
    Trailing,
}

/// Borrowed bytes ordered as strings decoded by [`String::from_utf8_lossy`].
struct LossyName<'a>(&'a [u8]);

impl LossyName<'_> {
    fn chars(&self) -> impl Iterator<Item = char> {
        self.0.utf8_chunks().flat_map(|chunk| {
            chunk
                .valid()
                .chars()
                .chain((!chunk.invalid().is_empty()).then_some(char::REPLACEMENT_CHARACTER))
        })
    }
}

impl Eq for LossyName<'_> {}

impl PartialEq for LossyName<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Ord for LossyName<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.0.is_ascii() && other.0.is_ascii() {
            self.0.cmp(other.0)
        } else {
            self.chars().cmp(other.chars())
        }
    }
}

impl PartialOrd for LossyName<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests;
