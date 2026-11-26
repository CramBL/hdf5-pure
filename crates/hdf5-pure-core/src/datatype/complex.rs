use core::num::NonZeroU32;

use super::layout::FloatingPointType;
use crate::FormatError;

/// A homogeneous rectangular complex datatype with a checked element size.
///
/// The real and imaginary components share one [`FloatingPointType`], including its layout
/// and byte order. Each complex element occupies twice the component size, within [`u32`]'s
/// range. This describes the native complex class, separate from [`crate::Datatype::Compound`]
/// conventions with named real and imaginary fields.
///
/// Class 11 of datatype message version 5 is defined in "The Datatype Message" of the
/// [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_dtmessage
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComplexType {
    component: FloatingPointType,
    size: NonZeroU32,
}

impl ComplexType {
    /// Constructs a complex datatype whose two components have the supplied floating-point type.
    ///
    /// # Errors
    ///
    /// Returns [`FormatError::ComplexSizeOverflow`] if twice the component size exceeds
    /// [`u32::MAX`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hdf5_pure_core::{ComplexType, Datatype, DatatypeByteOrder, FloatingPointType};
    ///
    /// let component = FloatingPointType::ieee754_binary32(DatatypeByteOrder::BigEndian);
    /// let complex = ComplexType::new(component)?;
    /// assert_eq!(complex.component(), component);
    /// assert_eq!(complex.size().get(), 8);
    /// assert_eq!(Datatype::Complex(complex).type_size(), 8);
    /// # Ok::<(), hdf5_pure_core::FormatError>(())
    /// ```
    pub fn new(component: FloatingPointType) -> Result<Self, FormatError> {
        let size = component
            .size()
            .get()
            .checked_mul(2)
            .and_then(NonZeroU32::new)
            .ok_or(FormatError::ComplexSizeOverflow {
                component_size: component.size().get(),
            })?;
        Ok(Self { component, size })
    }

    /// Returns the floating-point datatype shared by both components.
    pub const fn component(self) -> FloatingPointType {
        self.component
    }

    /// Returns the combined byte size of both components.
    pub const fn size(self) -> NonZeroU32 {
        self.size
    }
}
