//! Byte and datatype fixtures shared by datatype tests.
//!
//! "The Datatype Message" of the [format specification, version 4.0][spec] defines the
//! fixed-point and floating-point class bit fields and property layouts.
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage

use alloc::vec::Vec;

use byteorder::{ByteOrder, LittleEndian};
use hdf5_pure_core::{Datatype, DatatypeByteOrder, FixedPointType};

use super::super::header;
use super::super::message::atomic::{
    FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE, FLOAT_NORMALIZATION_SHIFT,
};
use super::super::message::version::DATATYPE_VERSION_ONE;

/// Builds a version 1 fixed-point message from raw size and bit range fields.
pub(super) fn build_fixed_point(
    size: u32,
    be: bool,
    signed: bool,
    bit_offset: u16,
    bit_precision: u16,
) -> Vec<u8> {
    let bf0 = if be { 0x01 } else { 0x00 } | if signed { 0x08 } else { 0x00 };
    let mut buf = header::build_datatype_header(0, DATATYPE_VERSION_ONE, [bf0, 0, 0], size);
    let mut props = [0u8; 4];
    LittleEndian::write_u16(&mut props[0..2], bit_offset);
    LittleEndian::write_u16(&mut props[2..4], bit_precision);
    buf.extend_from_slice(&props);
    buf
}

/// Builds a version 1 little-endian floating-point message with implied leading one.
///
/// Uses the full element width as the bit precision and the last bit as the sign bit.
///
/// # Panics
///
/// Panics if `size` is zero or the computed sign position exceeds [`u8::MAX`].
pub(super) fn build_float(
    size: u32,
    exp_loc: u8,
    exp_size: u8,
    mant_loc: u8,
    mant_size: u8,
    exp_bias: u32,
) -> Vec<u8> {
    let bf0 = FLOAT_NORMALIZATION_IMPLIED_LEADING_ONE << FLOAT_NORMALIZATION_SHIFT;
    let bf1 = u8::try_from(size * 8 - 1).unwrap();
    let bf2 = 0;
    let mut buf = header::build_datatype_header(1, DATATYPE_VERSION_ONE, [bf0, bf1, bf2], size);
    let mut props = [0u8; 12];
    LittleEndian::write_u16(&mut props[0..2], 0);
    LittleEndian::write_u16(&mut props[2..4], (size * 8) as u16);
    props[4] = exp_loc;
    props[5] = exp_size;
    props[6] = mant_loc;
    props[7] = mant_size;
    LittleEndian::write_u32(&mut props[8..12], exp_bias);
    buf.extend_from_slice(&props);
    buf
}

/// Returns the unsigned one-byte fixed-point datatype used in container fixtures.
pub(super) fn byte_member_type() -> Datatype {
    Datatype::FixedPoint(
        FixedPointType::new(1, DatatypeByteOrder::LittleEndian, false, 0, 8).unwrap(),
    )
}
