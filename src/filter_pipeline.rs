//! HDF5 Filter Pipeline message parsing (message type 0x000B).

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{string::String, string::ToString, vec, vec::Vec};

#[cfg(feature = "zfp")]
pub use h5_filter::FILTER_ZFP;
pub use h5_filter::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZF, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    H5Z_FLAG_OPTIONAL,
};

use crate::bytes::ensure_len;
use crate::error::FormatError;

/// Description of a single filter in a pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterDescription {
    /// Filter identification value.
    pub filter_id: u16,
    /// Optional filter name (required for filter_id >= 256 in v1).
    pub name: Option<String>,
    /// Bit 0 permits omission during output.
    pub flags: u16,
    /// Client data values passed to the filter.
    pub client_data: Vec<u32>,
}

impl FilterDescription {
    /// Returns whether this filter is marked optional for output.
    pub fn is_optional(&self) -> bool {
        h5_filter::FilterStep::optional(self)
    }
}

impl h5_filter::FilterStep for FilterDescription {
    fn id(&self) -> u16 {
        self.filter_id
    }
    fn flags(&self) -> u16 {
        self.flags
    }
    fn client_data(&self) -> &[u32] {
        &self.client_data
    }
}

/// A filter pipeline consisting of one or more filters.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterPipeline {
    /// Pipeline version (1 or 2).
    pub version: u8,
    /// Ordered list of filters.
    pub filters: Vec<FilterDescription>,
}

impl FilterPipeline {
    /// Parse a filter pipeline message from raw message bytes.
    pub fn parse(data: &[u8]) -> Result<FilterPipeline, FormatError> {
        ensure_len(data, 0, 2)?;
        let version = data[0];
        let number_of_filters = data[1] as usize;

        match version {
            1 => Self::parse_v1(data, number_of_filters),
            2 => Self::parse_v2(data, number_of_filters),
            _ => Err(FormatError::InvalidFilterPipelineVersion(version)),
        }
    }

    fn parse_v1(data: &[u8], number_of_filters: usize) -> Result<FilterPipeline, FormatError> {
        // version(1) + nfilters(1) + reserved(6) = 8 bytes header
        ensure_len(data, 0, 8)?;
        let mut pos = 8;
        let mut filters = Vec::with_capacity(number_of_filters);

        for _ in 0..number_of_filters {
            ensure_len(data, pos, 8)?;
            let filter_id = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let name_length = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
            let flags = u16::from_le_bytes([data[pos + 4], data[pos + 5]]);
            let num_client_data = u16::from_le_bytes([data[pos + 6], data[pos + 7]]) as usize;
            pos += 8;

            // Name (if present)
            let name = if name_length > 0 {
                ensure_len(data, pos, name_length)?;
                let name_bytes = &data[pos..pos + name_length];
                // Strip null terminator
                let name_str = core::str::from_utf8(
                    name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes),
                )
                .unwrap_or("")
                .to_string();
                // Pad to 8-byte boundary
                let padded = (name_length + 7) & !7;
                pos += padded;
                Some(name_str)
            } else {
                None
            };

            // Client data
            ensure_len(data, pos, num_client_data * 4)?;
            let mut client_data = Vec::with_capacity(num_client_data);
            for _ in 0..num_client_data {
                let val =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                client_data.push(val);
                pos += 4;
            }

            // Padding to 8-byte boundary if num_client_data is odd
            if !num_client_data.is_multiple_of(2) {
                pos += 4;
            }

            filters.push(FilterDescription {
                filter_id,
                name,
                flags,
                client_data,
            });
        }

        Ok(FilterPipeline {
            version: 1,
            filters,
        })
    }

    fn parse_v2(data: &[u8], number_of_filters: usize) -> Result<FilterPipeline, FormatError> {
        // version(1) + nfilters(1) = 2 bytes header (no reserved in v2)
        let mut pos = 2;
        let mut filters = Vec::with_capacity(number_of_filters);

        for _ in 0..number_of_filters {
            ensure_len(data, pos, 2)?;
            let filter_id = u16::from_le_bytes([data[pos], data[pos + 1]]);
            pos += 2;

            let name_length = if filter_id >= 256 {
                ensure_len(data, pos, 2)?;
                let nl = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                nl
            } else {
                0
            };

            ensure_len(data, pos, 4)?;
            let flags = u16::from_le_bytes([data[pos], data[pos + 1]]);
            let num_client_data = u16::from_le_bytes([data[pos + 2], data[pos + 3]]) as usize;
            pos += 4;

            let name = if name_length > 0 {
                ensure_len(data, pos, name_length)?;
                let name_bytes = &data[pos..pos + name_length];
                let name_str = core::str::from_utf8(
                    name_bytes.split(|&b| b == 0).next().unwrap_or(name_bytes),
                )
                .unwrap_or("")
                .to_string();
                pos += name_length;
                Some(name_str)
            } else {
                None
            };

            ensure_len(data, pos, num_client_data * 4)?;
            let mut client_data = Vec::with_capacity(num_client_data);
            for _ in 0..num_client_data {
                let val =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                client_data.push(val);
                pos += 4;
            }

            // No padding in v2

            filters.push(FilterDescription {
                filter_id,
                name,
                flags,
                client_data,
            });
        }

        Ok(FilterPipeline {
            version: 2,
            filters,
        })
    }

    /// Serialize the filter pipeline to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        match self.version {
            1 => self.serialize_v1(),
            2 => self.serialize_v2(),
            _ => vec![self.version, 0],
        }
    }

    fn serialize_v1(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(1); // version
        #[expect(
            clippy::cast_possible_truncation,
            reason = "filter count is written into the 1-byte number-of-filters field of the v1 pipeline message"
        )]
        buf.push(self.filters.len() as u8);
        buf.extend_from_slice(&[0u8; 6]); // reserved

        for f in &self.filters {
            buf.extend_from_slice(&f.filter_id.to_le_bytes());

            let name_bytes = match &f.name {
                Some(name) => {
                    let mut nb = name.as_bytes().to_vec();
                    nb.push(0); // null terminate
                    nb
                }
                None => Vec::new(),
            };
            #[expect(
                clippy::cast_possible_truncation,
                reason = "filter name length is written into the 2-byte name-length field of the v1 pipeline message"
            )]
            let name_length = name_bytes.len() as u16;
            buf.extend_from_slice(&name_length.to_le_bytes());
            buf.extend_from_slice(&f.flags.to_le_bytes());
            #[expect(
                clippy::cast_possible_truncation,
                reason = "client-data value count is written into the 2-byte client-data-values field of the v1 pipeline message"
            )]
            buf.extend_from_slice(&(f.client_data.len() as u16).to_le_bytes());

            if !name_bytes.is_empty() {
                buf.extend_from_slice(&name_bytes);
                // Pad to 8-byte boundary
                let padded = (name_bytes.len() + 7) & !7;
                let padding = padded - name_bytes.len();
                buf.extend_from_slice(&vec![0u8; padding]);
            }

            for &val in &f.client_data {
                buf.extend_from_slice(&val.to_le_bytes());
            }

            // Pad if odd number of client data values
            if f.client_data.len() % 2 != 0 {
                buf.extend_from_slice(&[0u8; 4]);
            }
        }

        buf
    }

    fn serialize_v2(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(2); // version
        #[expect(
            clippy::cast_possible_truncation,
            reason = "filter count is written into the 1-byte number-of-filters field of the v2 pipeline message"
        )]
        buf.push(self.filters.len() as u8);

        for f in &self.filters {
            buf.extend_from_slice(&f.filter_id.to_le_bytes());

            if f.filter_id >= 256 {
                let name_bytes = match &f.name {
                    Some(name) => {
                        let mut nb = name.as_bytes().to_vec();
                        nb.push(0);
                        nb
                    }
                    None => vec![0],
                };
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "filter name length is written into the 2-byte name-length field of the v2 pipeline message"
                )]
                buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(&f.flags.to_le_bytes());
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "client-data value count is written into the 2-byte client-data-values field of the v2 pipeline message"
                )]
                buf.extend_from_slice(&(f.client_data.len() as u16).to_le_bytes());
                buf.extend_from_slice(&name_bytes);
            } else {
                buf.extend_from_slice(&f.flags.to_le_bytes());
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "client-data value count is written into the 2-byte client-data-values field of the v2 pipeline message"
                )]
                buf.extend_from_slice(&(f.client_data.len() as u16).to_le_bytes());
            }

            for &val in &f.client_data {
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_v1_single_deflate() {
        // v1 pipeline with 1 filter: deflate (id=1) level 6
        let mut buf = vec![1u8, 1]; // version=1, nfilters=1
        buf.extend_from_slice(&[0u8; 6]); // reserved
        buf.extend_from_slice(&FILTER_DEFLATE.to_le_bytes()); // filter_id=1
        buf.extend_from_slice(&0u16.to_le_bytes()); // name_length
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&1u16.to_le_bytes()); // nclient
        buf.extend_from_slice(&6u32.to_le_bytes()); // level=6
        // odd client data count => 4 bytes padding
        buf.extend_from_slice(&[0u8; 4]);

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.version, 1);
        assert_eq!(fp.filters.len(), 1);
        assert_eq!(fp.filters[0].filter_id, FILTER_DEFLATE);
        assert_eq!(fp.filters[0].client_data, vec![6]);
        assert_eq!(fp.filters[0].name, None);
    }

    #[test]
    fn parse_v1_shuffle_and_deflate() {
        let mut buf = vec![1u8, 2];
        buf.extend_from_slice(&[0u8; 6]);
        // shuffle (id=2)
        buf.extend_from_slice(&FILTER_SHUFFLE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        // deflate (id=1)
        buf.extend_from_slice(&FILTER_DEFLATE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 4]); // padding for odd client data

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.filters.len(), 2);
        assert_eq!(fp.filters[0].filter_id, FILTER_SHUFFLE);
        assert_eq!(fp.filters[1].filter_id, FILTER_DEFLATE);
        assert_eq!(fp.filters[1].client_data, vec![6]);
    }

    #[test]
    fn parse_v2_deflate() {
        let mut buf = vec![2u8, 1]; // version=2, nfilters=1
        buf.extend_from_slice(&FILTER_DEFLATE.to_le_bytes()); // id=1
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&1u16.to_le_bytes()); // nclient
        buf.extend_from_slice(&6u32.to_le_bytes()); // level=6

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.version, 2);
        assert_eq!(fp.filters.len(), 1);
        assert_eq!(fp.filters[0].filter_id, FILTER_DEFLATE);
        assert_eq!(fp.filters[0].client_data, vec![6]);
    }

    #[test]
    fn parse_v2_three_filters() {
        let mut buf = vec![2u8, 3]; // version=2, nfilters=3
        // shuffle (id=2)
        buf.extend_from_slice(&FILTER_SHUFFLE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        // deflate (id=1)
        buf.extend_from_slice(&FILTER_DEFLATE.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes());
        // fletcher32 (id=3)
        buf.extend_from_slice(&FILTER_FLETCHER32.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes());

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.filters.len(), 3);
        assert_eq!(fp.filters[0].filter_id, FILTER_SHUFFLE);
        assert_eq!(fp.filters[1].filter_id, FILTER_DEFLATE);
        assert_eq!(fp.filters[2].filter_id, FILTER_FLETCHER32);
    }

    #[test]
    fn serialize_parse_roundtrip_v1() {
        let pipeline = FilterPipeline {
            version: 1,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_SHUFFLE,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![6],
                },
            ],
        };
        let serialized = pipeline.serialize();
        let parsed = FilterPipeline::parse(&serialized).unwrap();
        assert_eq!(parsed, pipeline);
    }

    #[test]
    fn serialize_parse_roundtrip_v2() {
        let pipeline = FilterPipeline {
            version: 2,
            filters: vec![
                FilterDescription {
                    filter_id: FILTER_DEFLATE,
                    name: None,
                    flags: 0,
                    client_data: vec![9],
                },
                FilterDescription {
                    filter_id: FILTER_FLETCHER32,
                    name: None,
                    flags: 0,
                    client_data: vec![],
                },
            ],
        };
        let serialized = pipeline.serialize();
        let parsed = FilterPipeline::parse(&serialized).unwrap();
        assert_eq!(parsed, pipeline);
    }

    #[test]
    fn custom_filter_with_name_v1() {
        let mut buf = vec![1u8, 1];
        buf.extend_from_slice(&[0u8; 6]);
        // custom filter: id=300 (>=256), name_length=11 ("myfilter\0" padded to 8)
        buf.extend_from_slice(&300u16.to_le_bytes());
        let name = b"myfilter\0"; // 9 bytes, pad to 16
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&2u16.to_le_bytes()); // nclient=2
        // name padded to 8-byte boundary: 9 bytes -> 16 bytes
        buf.extend_from_slice(name);
        buf.extend_from_slice(&[0u8; 7]); // pad 9->16
        // client data: 2 values (even, no padding needed)
        buf.extend_from_slice(&42u32.to_le_bytes());
        buf.extend_from_slice(&99u32.to_le_bytes());

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.filters[0].filter_id, 300);
        assert_eq!(fp.filters[0].name, Some("myfilter".to_string()));
        assert_eq!(fp.filters[0].client_data, vec![42, 99]);
    }

    #[test]
    fn custom_filter_with_name_v2() {
        let mut buf = vec![2u8, 1];
        // id=300 (>=256), so name_length field present
        buf.extend_from_slice(&300u16.to_le_bytes());
        let name = b"custom\0";
        buf.extend_from_slice(&(name.len() as u16).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // nclient=0
        buf.extend_from_slice(name);

        let fp = FilterPipeline::parse(&buf).unwrap();
        assert_eq!(fp.filters[0].filter_id, 300);
        assert_eq!(fp.filters[0].name, Some("custom".to_string()));
    }

    #[test]
    fn invalid_version() {
        let buf = vec![3u8, 0];
        let err = FilterPipeline::parse(&buf).unwrap_err();
        assert_eq!(err, FormatError::InvalidFilterPipelineVersion(3));
    }
}
