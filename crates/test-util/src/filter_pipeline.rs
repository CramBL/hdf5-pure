//! Independent filter pipeline message bodies.

/// Builds a version 2 pipeline with filter 40,500, named `custom`.
///
/// The client data values are 7 and the maximum 32-bit unsigned integer. The message is defined in
/// "The Data Storage - Filter Pipeline Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
pub fn user_filter(optional: bool) -> Vec<u8> {
    let mut pipeline = vec![VERSION_TWO, 1];
    let flags = if optional { FILTER_FLAG_OPTIONAL } else { 0 };
    pipeline.extend_from_slice(&40_500_u16.to_le_bytes());
    pipeline.extend_from_slice(&7_u16.to_le_bytes());
    pipeline.extend_from_slice(&flags.to_le_bytes());
    pipeline.extend_from_slice(&2_u16.to_le_bytes());
    pipeline.extend_from_slice(b"custom\0");
    pipeline.extend_from_slice(&7_u32.to_le_bytes());
    pipeline.extend_from_slice(&u32::MAX.to_le_bytes());
    pipeline
}

/// Version 2 of "The Data Storage - Filter Pipeline Message" in the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
const VERSION_TWO: u8 = 2;

/// The flag marking a filter optional in "The Data Storage - Filter Pipeline Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
const FILTER_FLAG_OPTIONAL: u16 = 0x0001;
