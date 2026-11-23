//! Independent filter pipeline message bodies.

/// Builds a version 2 pipeline with zero filters.
pub fn empty() -> Vec<u8> {
    Pipeline::new(Version::Two).build()
}

/// Builds a version 2 pipeline with filter 40,500, named `custom`.
///
/// The client data values are 7 and the maximum 32-bit unsigned integer. The message is defined in
/// "The Data Storage - Filter Pipeline Message" of the [format specification, version 4.0][spec].
///
/// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_filter
pub fn user_filter(optional: bool) -> Vec<u8> {
    let mut pipeline = Pipeline::new(Version::Two);
    let flags = if optional { FILTER_FLAG_OPTIONAL } else { 0 };
    pipeline
        .push(40_500, Some("custom"), flags, &[7, u32::MAX])
        .unwrap();
    pipeline.build()
}

/// The version of a Filter Pipeline fixture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Version {
    /// Version 1, with aligned names and padding after an odd number of client data values.
    One,
    /// Version 2, with unpadded names for filter identifiers of 256 or above.
    Two,
}

impl Version {
    /// Returns the stored version byte.
    pub const fn byte(self) -> u8 {
        match self {
            Self::One => VERSION_ONE,
            Self::Two => VERSION_TWO,
        }
    }
}

/// A fixture with checked filter records for one pipeline version.
pub struct Pipeline<'a> {
    version: Version,
    filters: Vec<Filter<'a>>,
}

impl<'a> Pipeline<'a> {
    /// Creates a pipeline with zero filters.
    pub fn new(version: Version) -> Self {
        Self {
            version,
            filters: Vec::new(),
        }
    }

    /// Appends a filter whose name and client data borrow from the caller.
    ///
    /// Returns [`None`] if the identifier is zero, the pipeline already holds 32 filters,
    /// or the client data count exceeds [`u16::MAX`]. For a present name, returns [`None`] if
    /// the identifier is below 256 in version 2, the name contains a null byte or non-ASCII
    /// text, or the encoded name length exceeds [`u16::MAX`].
    pub fn push(
        &mut self,
        id: u16,
        name: Option<&'a str>,
        flags: u8,
        client_data: &'a [u32],
    ) -> Option<()> {
        if id == RESERVED_FILTER_ID || self.filters.len() >= MAX_FILTERS {
            return None;
        }
        let name_length = if let Some(name) = name {
            if !self.stores_name_length(id) || !name.is_ascii() || name.contains('\0') {
                return None;
            }
            let length = name.len().checked_add(1)?;
            let length = match self.version {
                Version::One => length.checked_next_multiple_of(V1_NAME_ALIGNMENT)?,
                Version::Two => length,
            };
            u16::try_from(length).ok()?
        } else {
            0
        };
        let client_count = u16::try_from(client_data.len()).ok()?;
        self.filters.push(Filter {
            id,
            name,
            flags,
            client_data,
            name_length,
            client_count,
        });
        Some(())
    }

    /// Returns the pipeline's wire version.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Returns the filters in insertion order.
    pub fn filters(&self) -> &[Filter<'a>] {
        &self.filters
    }

    /// Builds the complete message body, including any required zero padding.
    pub fn build(&self) -> Vec<u8> {
        let filter_count = u8::try_from(self.filters.len()).unwrap();
        let mut bytes = vec![self.version.byte(), filter_count];
        if self.version == Version::One {
            bytes.extend_from_slice(&[0; V1_RESERVED_BYTES]);
        }
        for &Filter {
            id,
            name,
            flags,
            client_data,
            name_length,
            client_count,
        } in &self.filters
        {
            bytes.extend_from_slice(&id.to_le_bytes());
            if self.stores_name_length(id) {
                bytes.extend_from_slice(&name_length.to_le_bytes());
            }
            bytes.extend_from_slice(&u16::from(flags).to_le_bytes());
            bytes.extend_from_slice(&client_count.to_le_bytes());
            if let Some(name) = name {
                let end = bytes.len() + usize::from(name_length);
                bytes.extend_from_slice(name.as_bytes());
                bytes.resize(end, 0);
            }
            for &value in client_data {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            if self.version == Version::One && !client_count.is_multiple_of(2) {
                bytes.extend_from_slice(&0_u32.to_le_bytes());
            }
        }
        bytes
    }

    fn stores_name_length(&self, id: u16) -> bool {
        self.version == Version::One || id >= PREDEFINED_FILTER_LIMIT
    }
}

/// A filter whose stored lengths fit the fixture's version.
pub struct Filter<'a> {
    id: u16,
    name: Option<&'a str>,
    flags: u8,
    client_data: &'a [u32],
    name_length: u16,
    client_count: u16,
}

impl<'a> Filter<'a> {
    /// Returns the filter identifier stored in the record.
    pub fn id(&self) -> u16 {
        self.id
    }

    /// Returns the recorded name, without the terminator or padding.
    pub fn name(&self) -> Option<&'a str> {
        self.name
    }

    /// Returns the low eight definition flag bits stored in the record.
    pub fn flags(&self) -> u8 {
        self.flags
    }

    /// Returns the client values in the order supplied to [`Pipeline::push`].
    pub fn client_data(&self) -> &'a [u32] {
        self.client_data
    }
}

// Section `subsubsec_fmt4_dataobject_hdr_msg_filter`, version 4.0, defines these fields.
const VERSION_ONE: u8 = 1;
const VERSION_TWO: u8 = 2;
const MAX_FILTERS: usize = 32;
const RESERVED_FILTER_ID: u16 = 0;
const PREDEFINED_FILTER_LIMIT: u16 = 256;
const V1_RESERVED_BYTES: usize = 6;
const V1_NAME_ALIGNMENT: usize = 8;
const FILTER_FLAG_OPTIONAL: u8 = 1;
