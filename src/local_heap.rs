//! HDF5 Local Heap parsing.

#[cfg(not(feature = "std"))]
use alloc::string::String;

use crate::bytes::{read_length, read_offset};
use crate::convert::Narrow;
use crate::error::FormatError;
use crate::source::Source;

/// Parsed HDF5 Local Heap header.
#[derive(Debug, Clone)]
pub struct LocalHeap {
    /// Size of the data segment in bytes.
    pub data_segment_size: u64,
    /// Offset of the free list head within the data segment. Parsed for
    /// on-disk-format completeness; the reader does not free-list-allocate.
    #[allow(dead_code)]
    pub free_list_head_offset: u64,
    /// File address of the data segment.
    pub data_segment_address: u64,
}

impl LocalHeap {
    /// Parse a local heap header at the given offset in the file data.
    pub fn parse(
        file_data: &[u8],
        offset: usize,
        offset_size: u8,
        length_size: u8,
    ) -> Result<LocalHeap, FormatError> {
        // signature(4) + version(1) + reserved(3) = 8, then length_size*2 + offset_size
        let ls = length_size as usize;
        let os = offset_size as usize;
        let total = 8 + ls * 2 + os;
        // `offset` is a crafted file address; guard `offset + total` against a
        // `usize` overflow instead of wrapping past the bounds check (issue #140).
        let end = offset
            .checked_add(total)
            .ok_or(FormatError::UnexpectedEof {
                expected: usize::MAX,
                available: file_data.len(),
            })?;
        if end > file_data.len() {
            return Err(FormatError::UnexpectedEof {
                expected: end,
                available: file_data.len(),
            });
        }

        if &file_data[offset..offset + 4] != b"HEAP" {
            return Err(FormatError::InvalidLocalHeapSignature);
        }

        let version = file_data[offset + 4];
        if version != 0 {
            return Err(FormatError::InvalidLocalHeapVersion(version));
        }

        let mut pos = offset + 8;
        let data_segment_size = read_length(file_data, pos, length_size)?;
        pos += ls;
        let free_list_head_offset = read_length(file_data, pos, length_size)?;
        pos += ls;
        let data_segment_address = read_offset(file_data, pos, offset_size)?;

        Ok(LocalHeap {
            data_segment_size,
            free_list_head_offset,
            data_segment_address,
        })
    }

    /// Read a null-terminated string from the heap's data segment at the given byte offset.
    pub fn read_string(&self, file_data: &[u8], string_offset: u64) -> Result<String, FormatError> {
        let seg_addr = self.data_segment_address.to_usize()?;
        let str_start =
            seg_addr
                .checked_add(string_offset.to_usize()?)
                .ok_or(FormatError::OffsetOverflow {
                    offset: self.data_segment_address,
                    length: string_offset,
                })?;
        let seg_end = seg_addr
            .checked_add(self.data_segment_size.to_usize()?)
            .ok_or(FormatError::OffsetOverflow {
                offset: self.data_segment_address,
                length: self.data_segment_size,
            })?;

        if str_start >= file_data.len() || str_start >= seg_end {
            return Err(FormatError::UnexpectedEof {
                // `str_start` is a crafted address and can be near `usize::MAX`;
                // this diagnostic must not overflow (issue #140).
                expected: str_start.saturating_add(1),
                available: file_data.len(),
            });
        }

        // Find null terminator
        let search_end = seg_end.min(file_data.len());
        let mut end = str_start;
        while end < search_end && file_data[end] != 0 {
            end += 1;
        }

        if end >= search_end {
            return Err(FormatError::UnexpectedEof {
                expected: end + 1,
                available: search_end,
            });
        }

        let s = core::str::from_utf8(&file_data[str_start..end]).map_err(|source| {
            FormatError::InvalidLocalHeapName {
                offset: string_offset,
                source,
            }
        })?;
        Ok(String::from(s))
    }

    /// Parse a local heap header from a [`Source`] (small bounded window).
    pub fn parse_from_source<S: Source + ?Sized>(
        source: &S,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<LocalHeap, FormatError> {
        // signature(4) + version(1) + reserved(3) + length_size*2 + offset_size
        let total = 8 + (length_size as usize) * 2 + offset_size as usize;
        let buf = source.read_metadata_at(address, total)?;
        Self::parse(&buf, 0, offset_size, length_size)
    }

    /// Read a null-terminated string from a pre-fetched copy of the heap's data
    /// segment (the `data_segment_size` bytes starting at
    /// [`data_segment_address`](Self::data_segment_address)).
    ///
    /// This is the streaming counterpart of [`read_string`](Self::read_string):
    /// the caller reads the data segment once via [`Source`] and slices every
    /// name out of it, so a group with many links costs one segment read rather
    /// than one read per name.
    pub fn read_string_in_segment(
        &self,
        segment: &[u8],
        string_offset: u64,
    ) -> Result<String, FormatError> {
        let start = string_offset.to_usize()?;
        let seg_len = self.data_segment_size.to_usize()?;
        let search_end = seg_len.min(segment.len());

        if start >= search_end {
            return Err(FormatError::UnexpectedEof {
                // `start` is a crafted string offset and can be near `usize::MAX`;
                // this diagnostic must not overflow (issue #140).
                expected: start.saturating_add(1),
                available: search_end,
            });
        }

        let mut end = start;
        while end < search_end && segment[end] != 0 {
            end += 1;
        }

        if end >= search_end {
            return Err(FormatError::UnexpectedEof {
                expected: end + 1,
                available: search_end,
            });
        }

        let s = core::str::from_utf8(&segment[start..end]).map_err(|source| {
            FormatError::InvalidLocalHeapName {
                offset: string_offset,
                source,
            }
        })?;
        Ok(String::from(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_heap_file(
        heap_offset: usize,
        data_seg_offset: usize,
        strings: &[&str],
        offset_size: u8,
        length_size: u8,
    ) -> Vec<u8> {
        // Build data segment
        let mut data_seg = Vec::new();
        for s in strings {
            data_seg.extend_from_slice(s.as_bytes());
            data_seg.push(0); // null terminator
        }
        let data_seg_size = data_seg.len();

        let total_size = data_seg_offset + data_seg_size + 64;
        let mut file = vec![0u8; total_size];

        // Write heap header at heap_offset
        let mut pos = heap_offset;
        file[pos..pos + 4].copy_from_slice(b"HEAP");
        pos += 4;
        file[pos] = 0; // version
        pos += 1;
        // reserved 3
        pos += 3;
        // data_segment_size
        write_val(&mut file, pos, data_seg_size as u64, length_size);
        pos += length_size as usize;
        // free_list_head_offset
        write_val(&mut file, pos, 0xFFFFFFFF, length_size);
        pos += length_size as usize;
        // data_segment_address
        write_val(&mut file, pos, data_seg_offset as u64, offset_size);

        // Write data segment
        file[data_seg_offset..data_seg_offset + data_seg_size].copy_from_slice(&data_seg);

        file
    }

    fn write_val(buf: &mut [u8], pos: usize, val: u64, size: u8) {
        match size {
            4 => buf[pos..pos + 4].copy_from_slice(&(val as u32).to_le_bytes()),
            8 => buf[pos..pos + 8].copy_from_slice(&val.to_le_bytes()),
            _ => panic!("test"),
        }
    }

    #[test]
    fn parse_heap_header() {
        let file = build_heap_file(0, 100, &["hello", "world"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        assert_eq!(heap.data_segment_address, 100);
        assert_eq!(heap.data_segment_size, 12); // "hello\0world\0"
    }

    #[test]
    fn read_string_at_offset_0() {
        let file = build_heap_file(0, 100, &["hello", "world"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        let s = heap.read_string(&file, 0).unwrap();
        assert_eq!(s, "hello");
    }

    #[test]
    fn read_string_at_offset_6() {
        let file = build_heap_file(0, 100, &["hello", "world"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        let s = heap.read_string(&file, 6).unwrap();
        assert_eq!(s, "world");
    }

    #[test]
    fn invalid_signature() {
        let mut file = build_heap_file(0, 100, &["x"], 8, 8);
        file[0] = b'X';
        let err = LocalHeap::parse(&file, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLocalHeapSignature);
    }

    // The third byte of "hello" is 0xFF, so the decode is valid up to 2.
    #[test]
    fn read_string_rejects_a_name_that_is_not_utf8() {
        let mut file = build_heap_file(0, 100, &["hello"], 8, 8);
        file[102] = 0xFF;
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();

        let err = heap.read_string(&file, 0).unwrap_err();
        let FormatError::InvalidLocalHeapName { offset, source } = err else {
            panic!("expected InvalidLocalHeapName, got {err:?}");
        };
        assert_eq!(offset, 0);
        assert_eq!(source.valid_up_to(), 2);

        let err = heap.read_string_in_segment(&file[100..], 0).unwrap_err();
        assert!(
            matches!(err, FormatError::InvalidLocalHeapName { offset: 0, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn read_string_past_segment() {
        let file = build_heap_file(0, 100, &["hi"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        let err = heap.read_string(&file, 100).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn parse_heap_4byte_offsets() {
        let file = build_heap_file(0, 80, &["test"], 4, 4);
        let heap = LocalHeap::parse(&file, 0, 4, 4).unwrap();
        assert_eq!(heap.data_segment_address, 80);
        let s = heap.read_string(&file, 0).unwrap();
        assert_eq!(s, "test");
    }

    #[test]
    fn invalid_version() {
        let mut file = build_heap_file(0, 100, &["x"], 8, 8);
        file[4] = 1; // bad version
        let err = LocalHeap::parse(&file, 0, 8, 8).unwrap_err();
        assert_eq!(err, FormatError::InvalidLocalHeapVersion(1));
    }

    #[test]
    fn parse_offset_near_usize_max_errors_without_overflow() {
        // A crafted heap address near `usize::MAX` must yield an EOF error, not
        // panic on `offset + total` overflowing `usize` (issue #140).
        let file = build_heap_file(0, 80, &["test"], 8, 8);
        let err = LocalHeap::parse(&file, usize::MAX - 3, 8, 8).unwrap_err();
        assert!(matches!(err, FormatError::UnexpectedEof { .. }));
    }

    #[test]
    fn read_string_offset_near_usize_max_errors_without_overflow() {
        // A crafted string offset near `usize::MAX` must not overflow the
        // diagnostic `start + 1` in either read path (issue #140).
        let file = build_heap_file(0, 80, &["test"], 8, 8);
        let heap = LocalHeap::parse(&file, 0, 8, 8).unwrap();
        assert!(heap.read_string(&file, u64::MAX).is_err());
        assert!(heap.read_string_in_segment(&file, u64::MAX).is_err());
    }
}
