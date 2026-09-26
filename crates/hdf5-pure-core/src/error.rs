use alloc::collections::TryReserveError;
use alloc::string::String;
use core::fmt;
use core::num::NonZeroUsize;
use core::str::Utf8Error;

/// Reports format, filter, and runtime failures through the `hdf5-pure` API.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// The HDF5 magic signature was not found at any valid offset.
    SignatureNotFound,
    /// The superblock version is not supported.
    UnsupportedVersion(u8),
    /// Unexpected end of data.
    UnexpectedEof {
        /// Number of bytes expected.
        expected: usize,
        /// Number of bytes actually available.
        available: usize,
    },
    /// Invalid offset size (must be 2, 4, or 8).
    InvalidOffsetSize(u8),
    /// Invalid length size (must be 2, 4, or 8).
    InvalidLengthSize(u8),
    /// Invalid object header signature.
    InvalidObjectHeaderSignature,
    /// Invalid object header version.
    InvalidObjectHeaderVersion(u8),
    /// A version 1 object-header message has a data size that is not a multiple of eight.
    ///
    /// Contains the declared message data size.
    InvalidObjectHeaderMessageSize(u16),
    /// Unknown message type that is marked as must-understand.
    UnsupportedMessage(u16),
    /// Invalid datatype class.
    InvalidDatatypeClass(u8),
    /// Invalid datatype version for a given class.
    InvalidDatatypeVersion {
        /// The type class.
        class: u8,
        /// The version found.
        version: u8,
    },
    /// A parsed datatype occupies zero bytes per element.
    ZeroSizedDatatype {
        /// The type class that declared it.
        class: u8,
    },
    /// Invalid string padding type.
    InvalidStringPadding(u8),
    /// Invalid character set.
    InvalidCharacterSet(u8),
    /// Invalid byte order.
    InvalidByteOrder(u8),
    /// Invalid reference type.
    InvalidReferenceType(u8),
    /// Invalid file-space management strategy code in a File Space Info message.
    InvalidFileSpaceStrategy(u8),
    /// Unsupported File Space Info message version (only version 1 is handled).
    UnsupportedFileSpaceInfoVersion(u8),
    /// A paged file-space strategy was requested with a page size the writer
    /// cannot use: it must be a power of two of at least 512 bytes.
    InvalidFileSpacePageSize(u64),
    /// A paged file-space strategy was requested alongside a userblock that is
    /// not a whole number of pages. File-space pages are measured from the file
    /// base, so the two boundaries coincide only when the userblock divides by
    /// the page size: `(userblock bytes, page size)`.
    UserblockNotPageAligned(u64, u64),
    /// A userblock size the format does not define: it must be zero, or a power
    /// of two of at least 512 bytes. A reader looks for the superblock at 0, 512,
    /// 1024, and so on doubling, so any other size produces a file nothing can
    /// open.
    InvalidUserblockSize(u64),
    /// More userblock content was supplied than the userblock region holds. The
    /// overflow would displace the superblock, so the writer rejects the content.
    UserblockContentTooLarge {
        /// Bytes supplied.
        content: u64,
        /// Bytes the userblock region holds.
        userblock: u64,
    },
    /// A free-space manager block (`FSHD`/`FSSE`) is malformed.
    InvalidFreeSpaceManager,
    /// An enumeration datatype was built over a base type that is not an
    /// integer. HDF5 enumerations must have a fixed-point base.
    EnumBaseNotInteger,
    /// An enumeration member's value does not occupy exactly the base type's
    /// size: `(member name, expected bytes, actual bytes)`.
    EnumMemberValueSize(String, u32, usize),
    /// An enumeration member's integer value does not fit in the base type:
    /// `(member name, value, base size in bytes)`.
    EnumMemberValueRange(String, i64, u32),
    /// A compound datatype has a zero total size.
    InvalidCompoundSize,
    /// A compound datatype contains no fields.
    EmptyCompoundType,
    /// A compound datatype contains the same field name more than once.
    DuplicateCompoundField(String),
    /// A compound field extends past the declared compound size.
    CompoundFieldOutOfBounds {
        /// Field name.
        name: String,
        /// Field byte offset.
        offset: u64,
        /// Field size in bytes.
        field_size: u32,
        /// Declared compound size in bytes.
        compound_size: u32,
    },
    /// Two compound fields overlap.
    CompoundFieldOverlap {
        /// Earlier field in byte order.
        first: String,
        /// Later field in byte order.
        second: String,
    },
    /// A named compound field was not present.
    CompoundFieldMissing(String),
    /// A compound field has an incompatible datatype.
    CompoundFieldTypeMismatch(String),
    /// Invalid dataspace version.
    InvalidDataspaceVersion(u8),
    /// Invalid dataspace type.
    InvalidDataspaceType(u8),
    /// Invalid data layout version.
    InvalidLayoutVersion(u8),
    /// Invalid data layout class.
    InvalidLayoutClass(u8),
    /// A chunked layout's chunk indexing type is outside the five the Chunk
    /// Indexing Type table of "The Data Layout Message" defines, in the [format
    /// specification, version 4.0][spec]: Single Chunk (1), Implicit (2), Fixed
    /// Array (3), Extensible Array (4), and version 2 B-tree (5).
    /// `H5O__layout_decode` rejects the same bytes (`H5Olayout.c`, HDF5
    /// 1.14.6).
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
    InvalidChunkIndexType(u8),
    /// The dataset's Fill Value message could not be parsed, and a read needed
    /// it: part of the dataset's storage was never allocated, so what those
    /// elements read as is undetermined. A dataset whose storage is fully
    /// allocated reads normally regardless of this message.
    UnreadableFillValue,
    /// Type mismatch when reading data.
    TypeMismatch {
        /// Expected type description.
        expected: &'static str,
        /// Actual type description.
        actual: &'static str,
    },
    /// Data size mismatch.
    DataSizeMismatch {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size in bytes.
        actual: usize,
    },
    /// Invalid local heap signature.
    InvalidLocalHeapSignature,
    /// Invalid local heap version.
    InvalidLocalHeapVersion(u8),
    /// A link name in a group's local heap is not UTF-8. A version 1 group
    /// stores the names of its links in a local heap, defined in "Local Heaps"
    /// of the [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_localheap
    InvalidLocalHeapName {
        /// The name's offset within the heap's data segment.
        offset: u64,
        /// The decode failure.
        source: Utf8Error,
    },
    /// Invalid B-tree v1 signature.
    InvalidBTreeSignature,
    /// Invalid B-tree node type.
    InvalidBTreeNodeType(u8),
    /// Invalid symbol table node signature.
    InvalidSymbolTableNodeSignature,
    /// Invalid symbol table node version.
    InvalidSymbolTableNodeVersion(u8),
    /// Path not found during group traversal.
    PathNotFound(String),
    /// Invalid Link message version.
    InvalidLinkVersion(u8),
    /// Invalid link type code.
    InvalidLinkType(u8),
    /// Invalid Link Info message version.
    InvalidLinkInfoVersion(u8),
    /// Invalid B-tree v2 signature.
    InvalidBTreeV2Signature,
    /// Invalid B-tree v2 version.
    InvalidBTreeV2Version(u8),
    /// Invalid fractal heap signature.
    InvalidFractalHeapSignature,
    /// Invalid fractal heap version.
    InvalidFractalHeapVersion(u8),
    /// Invalid heap ID type.
    InvalidHeapIdType(u8),
    /// A fractal-heap "huge" object's heap ID referenced a B-tree key that is
    /// not present in the heap's huge-objects v2 B-tree.
    HugeObjectNotFound(u64),
    /// A fractal heap's huge-objects v2 B-tree is not the indirectly accessed,
    /// non-filtered layout (record type 1) this reader decodes: either the tree
    /// declares a different record type, or its records are too short to hold
    /// that one. Reading its records as that layout would decode an object ID
    /// out of another field's bytes.
    UnexpectedHugeObjectBTree {
        /// The record type the B-tree declares.
        tree_type: u8,
        /// The record size the B-tree declares, in bytes.
        record_size: usize,
        /// The bytes a type-1 record needs: address + length + object ID.
        required: usize,
    },
    /// A fractal-heap object lives in an I/O-filter-encoded heap (filtered
    /// managed or huge storage), whose filtered bytes this reader does not
    /// decode. Link and attribute heaps are never filtered, so this does not
    /// arise for them.
    UnsupportedFilteredHeapObject,
    /// A dataset uses the Virtual (VDS) data layout, which maps its elements to
    /// regions of other datasets, possibly in other files. The reader rejects a
    /// dataset whose virtual mappings it cannot resolve.
    UnsupportedVirtualLayout,
    /// A dataset's element bytes live in files outside this one
    /// (`H5Pset_external`, the External Data Files header message). Its layout
    /// message is contiguous with an undefined data address, the same encoding
    /// used by an unwritten dataset. The reader rejects external storage because
    /// it cannot follow the external files.
    UnsupportedExternalStorage,
    /// Invalid attribute message version.
    InvalidAttributeVersion(u8),
    /// Invalid Attribute Info message version.
    InvalidAttributeInfoVersion(u8),
    /// Invalid shared message version.
    InvalidSharedMessageVersion(u8),
    /// An attribute message's flags byte set a bit the format does not define.
    /// Only bit 0 (shared datatype) and bit 1 (shared dataspace) exist, so any
    /// other bit means the message is not what it claims to be.
    InvalidAttributeFlags(u8),
    /// A message body references the shared object header message (SOHM) heap,
    /// but parsing lacks a shared-message table. The superblock extension may
    /// omit the table, the table may be unreadable, or the caller may parse a
    /// message body without its file. A committed (`H5Tcommit`) datatype
    /// references another object header and is resolved separately.
    UnsupportedSohmReference,
    /// A file's shared-message table, or a Shared Message Table message identifying
    /// it, declares a version this format does not define.
    InvalidSohmTableVersion(u8),
    /// A Shared Message Table message declares no indexes, or more than the
    /// eight the format allows. The count sizes the table's own read, so a
    /// wrong one reads neighbouring bytes as index headers.
    InvalidSohmIndexCount(u8),
    /// The shared-message table does not start with its `SMTB` signature.
    InvalidSohmTableSignature,
    /// A shared-message list index does not start with its `SMLI` signature.
    InvalidSohmListSignature,
    /// A shared-message index header declares a storage kind that is neither a
    /// list (0) nor a version 2 B-tree (1).
    InvalidSohmIndexKind(u8),
    /// A shared-message index refers to a version 2 B-tree with a type other
    /// than shared-message index (type 7). Contains the B-tree type.
    InvalidSohmBTreeType(u8),
    /// A shared-message index record identifies a storage location that is neither
    /// the heap (0) nor an object header (1).
    InvalidSohmRecordLocation(u8),
    /// A message body references the shared-message heap without a matching
    /// index that covers its type and has an allocated heap. Contains the raw
    /// type ID of the referenced message.
    SohmIndexMissing(u16),
    /// A shared message reference identifies an object header without a message
    /// of the referenced type.
    SharedMessageMissing {
        /// Address of the object header identified by the reference.
        object_header_address: u64,
        /// Raw type ID of the message the reference stood in for.
        message_type: u16,
    },
    /// A message body holds a reference to a shared message, and the parse that
    /// met it had no access to the file the reference addresses. Carries the raw
    /// type ID of the referenced message.
    UnresolvedSharedMessage(u16),
    /// A dataset or attribute refers to a committed datatype at a path where the file
    /// being written places no such object.
    UnknownCommittedDatatype(String),
    /// A dataset or attribute refers to a committed datatype whose encoding differs
    /// from its own. The two would disagree about how to read the element bytes,
    /// and the committed one is what every reader would believe.
    CommittedDatatypeMismatch {
        /// Path of the referenced committed datatype object.
        path: String,
        /// Name of the dataset or attribute that named it.
        user: String,
    },
    /// Invalid global heap collection signature.
    InvalidGlobalHeapSignature,
    /// Invalid global heap version.
    InvalidGlobalHeapVersion(u8),
    /// Global heap object not found.
    GlobalHeapObjectNotFound {
        /// Address of the collection.
        collection_address: u64,
        /// Index that was not found.
        index: u16,
    },
    /// Variable-length data error.
    VlDataError(String),
    /// A variable-length read exceeded its configured element limit.
    VariableLengthElementLimitExceeded {
        /// Maximum number of elements permitted by the caller.
        limit: usize,
        /// Number of elements present in the selected data.
        actual: u64,
    },
    /// A variable-length read exceeded its configured payload-byte limit.
    VariableLengthByteLimitExceeded {
        /// Maximum number of payload bytes permitted by the caller.
        limit: usize,
        /// Number of payload bytes required by the selected data.
        required: u64,
    },
    /// Serialization error.
    SerializationError(String),
    /// A name a builder was given for a dataset, a group, or a committed datatype is not a
    /// component of an object path: the empty name, `.`, or a name holding `/`. The object
    /// under such a link has no path, and the write reports the name it rejects.
    InvalidLinkName(String),
    /// Dataset is missing data.
    DatasetMissingData,
    /// Dataset is missing shape.
    DatasetMissingShape,
    /// The dataset's element count implied by its shape does not match the
    /// amount of data supplied (`shape.product() * element_size != data.len()`).
    ShapeDataMismatch {
        /// Number of data bytes the shape requires (`product(shape) * element_size`).
        expected: usize,
        /// Number of data bytes actually supplied.
        actual: usize,
        /// Size in bytes of one element (the dataset's datatype size), used to
        /// report the mismatch in elements as well as bytes. Non-zero by type,
        /// so the division in the message is well defined.
        element_size: NonZeroUsize,
    },
    /// A chunked, filtered, or extensible dataset's chunk geometry is invalid: for
    /// example chunk dimensions whose rank disagrees with the shape, a zero chunk
    /// dimension, a maximum shape whose rank disagrees with the shape or that is
    /// smaller than the current shape, or chunking requested on a scalar dataset.
    /// The writer rejects malformed geometry before splitting chunks. The payload
    /// is a human-readable reason.
    InvalidChunkGeometry(&'static str),
    /// Invalid filter pipeline version.
    InvalidFilterPipelineVersion(u8),
    /// Unsupported filter ID.
    UnsupportedFilter(u16),
    /// Filter processing error, including a stream that failed to decode. Every
    /// filter in the pipeline, including shuffle, scale-offset, LZF, and deflate,
    /// reports a bad chunk with this variant. The payload identifies the filter.
    FilterError(String),
    /// Fletcher32 checksum mismatch.
    Fletcher32Mismatch {
        /// Expected checksum.
        expected: u32,
        /// Computed checksum.
        computed: u32,
    },
    /// Chunked dataset read error.
    ChunkedReadError(String),
    /// CRC32C checksum mismatch.
    ChecksumMismatch {
        /// The checksum stored in the file.
        expected: u32,
        /// The checksum we computed.
        computed: u32,
    },
    /// Maximum nesting/continuation depth exceeded (malformed data protection).
    NestingDepthExceeded,
    /// ZFP filter configuration is invalid (e.g. missing element type, rank out of range).
    UnsupportedZfp(String),
    /// A file-derived 64-bit value (an offset, length, size, or element count)
    /// does not fit in the target integer type on this platform. This is the
    /// checked conversion for file-derived values. On a 32-bit host, `usize` is
    /// 32 bits, so an HDF5 offset or length above `usize::MAX` exceeds its range.
    /// The error retains the original value, and `target` identifies the integer
    /// type (for example, `"usize"` or `"u32"`).
    ValueTooLargeForPlatform {
        /// The original 64-bit value read from the file.
        value: u64,
        /// The platform integer type the value could not fit into.
        target: &'static str,
    },
    /// Two file-derived values (typically an offset and a length) overflow `u64`
    /// when added to form a slice bound. The reader reports the overflow.
    OffsetOverflow {
        /// First operand (typically the base offset/address).
        offset: u64,
        /// Second operand (typically the length/size).
        length: u64,
    },
    /// An absolute file position lies below the superblock's base address, so it
    /// lies in the userblock, where HDF5 structures cannot live. The reader
    /// reports the position and the base address.
    AddressBelowBase {
        /// The absolute file position that could not be made base-relative.
        address: u64,
        /// The superblock base address it was below.
        base: u64,
    },
    /// An allocation sized from the file failed. A typed whole-dataset read
    /// reserves its output buffer once with `try_reserve`, for as many values as
    /// the read produces, and reports the allocator's failure here.
    AllocationFailed {
        /// The number of values the reservation requested.
        values: usize,
        /// The allocator's reason for the failure.
        source: TryReserveError,
    },
    /// A random-access byte source failed to supply bytes. The string carries a backend-specific reason
    /// (e.g. an underlying `std::io::Error` rendered to text), so this stays
    /// `no_std`/`alloc`-friendly and free of an `std::io` dependency.
    Source(String),
    /// The library-version bounds passed to `FileBuilder::with_libver_bounds`
    /// admit no format the writer produces. The writer supports the 1.8 and 1.10
    /// formats, so an upper bound older than 1.8, or one below the lower bound,
    /// is unsatisfiable. The fields hold the default format and the bounds as
    /// library-version labels.
    LibverBoundsUnsatisfiable {
        /// The library-version label of the format this crate writes by default.
        writes: &'static str,
        /// The lower bound.
        requested_low: &'static str,
        /// The upper bound.
        requested_high: &'static str,
    },
    /// The file's content needs a newer on-disk format than its library-version
    /// bounds allow. `content` identifies the feature that requires the newer
    /// format, and `needs` holds the required library-version label.
    LibverTooOldForContent {
        /// What in the file requires a newer format, e.g. `"a chunked dataset"`.
        content: &'static str,
        /// The library-version label of the format that content requires.
        needs: &'static str,
        /// The library-version label of the format the bounds resolved to.
        writing: &'static str,
    },
    /// An HDF5 object reference (`H5R_OBJECT`) could not be resolved to an
    /// object: the stored address is null or undefined (`HADDR_UNDEF`), or it
    /// does not point at a group or dataset object header. The payload is the
    /// stored (base-relative) address, preserved for diagnostics.
    InvalidObjectReference(u64),
    /// A Fill Value message (`0x0005`) has an on-disk version this crate does
    /// not recognize (only 1, 2, and 3 are defined). The payload is the version
    /// byte found.
    UnsupportedFillValueVersion(u8),
    /// A user-supplied fill value's byte width does not match the dataset's
    /// datatype element size (for example a `u8` fill value on an `i32`
    /// dataset). The fields carry the datatype element size and the fill value
    /// size, both in bytes.
    FillValueSizeMismatch {
        /// The dataset datatype's element size in bytes.
        expected: usize,
        /// The supplied fill value's size in bytes.
        actual: usize,
    },
    /// An attribute's serialized message exceeds the version 2 object header's
    /// 2-byte message-size field. The writer rejects an attribute that cannot
    /// fit in compact storage. The fields hold the attribute name and its
    /// serialized message size. The limit is [`OBJECT_HEADER_MESSAGE_MAX`].
    AttributeMessageTooLarge {
        /// The attribute's name.
        name: String,
        /// The attribute message's serialized size in bytes.
        size: usize,
    },
    /// An object-header message exceeds the version 2 object header's 2-byte
    /// message-size field. The whole-file writer reports the message type code
    /// and the serialized size. The in-place editor reports `Error::EditUnsupported`
    /// for the same condition. The limit is [`OBJECT_HEADER_MESSAGE_MAX`].
    ObjectHeaderMessageTooLarge {
        /// The header message's type code (see the HDF5 message type table).
        message_type: u16,
        /// The message's serialized size in bytes.
        size: usize,
    },
    /// An attribute's name, datatype, or dataspace exceeds its 2-byte length
    /// field in the attribute message. The bound applies to the attribute's
    /// description. Attribute data may be stored in dense storage as a
    /// fractal-heap huge object. The `field` value identifies the length field.
    AttributeFieldTooLong {
        /// The attribute's name.
        name: String,
        /// Which 2-byte length field overflowed: `"name"`, `"datatype"`, or
        /// `"dataspace"`.
        field: &'static str,
        /// That field's encoded length in bytes.
        size: usize,
        /// The largest length the message can describe, in bytes.
        limit: usize,
    },
    /// An object's attributes need more space than a dense attribute heap can
    /// address: its offsets are 40 bits wide, so the blocks holding the
    /// attributes cannot span more than `limit` bytes between them. Reaching this
    /// takes about a terabyte of attributes on a single object.
    ///
    /// A set that exceeds the host's address range reports
    /// [`FormatError::ValueTooLargeForPlatform`].
    DenseAttributeHeapTooLarge {
        /// The heap address space, in bytes.
        limit: u64,
    },
    /// A caller passed zero as the declared width of a fixed-width string.
    /// HDF5 string datatypes require a positive width. Explicitly sized dataset
    /// and attribute strings can report this error. A width derived from empty
    /// values is one byte.
    ZeroFixedStringWidth,
    /// An element passed to a fixed-width string dataset or attribute exceeds
    /// its declared width. The writer rejects the element.
    FixedStringTooLong {
        /// Position of the offending element among the values passed. Zero for
        /// a scalar attribute, which holds one value.
        index: usize,
        /// That element's length in bytes.
        len: usize,
        /// The declared width, in bytes.
        width: u32,
    },
    /// A numeric element in a file is wider than the 64-bit word the typed
    /// numeric readers model an element as.
    ///
    /// Typed numeric readers assemble an element from its leading eight bytes.
    /// They reject wider elements because those bytes hold only part of a value.
    /// Raw dataset reads remain available, and attribute datatype reads report
    /// the stored type.
    NumericElementTooWide {
        /// Storage width of one element, in bytes.
        size: usize,
    },
}

pub const OBJECT_HEADER_MESSAGE_MAX: usize = u16::MAX as usize;

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureNotFound => {
                write!(f, "HDF5 signature not found at any valid offset")
            }
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported superblock version: {v}")
            }
            Self::UnexpectedEof {
                expected,
                available,
            } => {
                write!(f, "unexpected EOF: need {expected} bytes, have {available}")
            }
            Self::InvalidOffsetSize(s) => {
                write!(f, "invalid offset size: {s} (must be 2, 4, or 8)")
            }
            Self::InvalidLengthSize(s) => {
                write!(f, "invalid length size: {s} (must be 2, 4, or 8)")
            }
            Self::InvalidObjectHeaderSignature => {
                write!(f, "invalid object header signature")
            }
            Self::InvalidObjectHeaderVersion(v) => {
                write!(f, "invalid object header version: {v}")
            }
            Self::InvalidObjectHeaderMessageSize(size) => {
                write!(
                    f,
                    "invalid version 1 object-header message size {size}: size must be a multiple of eight"
                )
            }
            Self::UnsupportedMessage(id) => {
                write!(
                    f,
                    "unsupported message type {id:#06x} marked as must-understand"
                )
            }
            Self::InvalidDatatypeClass(c) => {
                write!(f, "invalid datatype class: {c}")
            }
            Self::InvalidDatatypeVersion { class, version } => {
                write!(f, "invalid datatype version {version} for class {class}")
            }
            Self::ZeroSizedDatatype { class } => {
                write!(
                    f,
                    "datatype class {class} declares a zero-byte element size"
                )
            }
            Self::InvalidStringPadding(p) => {
                write!(f, "invalid string padding type: {p}")
            }
            Self::InvalidCharacterSet(c) => {
                write!(f, "invalid character set: {c}")
            }
            Self::InvalidByteOrder(b) => {
                write!(f, "invalid byte order: {b}")
            }
            Self::InvalidReferenceType(r) => {
                write!(f, "invalid reference type: {r}")
            }
            Self::InvalidFileSpaceStrategy(s) => {
                write!(f, "invalid file-space strategy code: {s}")
            }
            Self::UnsupportedFileSpaceInfoVersion(v) => {
                write!(f, "unsupported File Space Info message version: {v}")
            }
            Self::InvalidFileSpacePageSize(p) => {
                write!(
                    f,
                    "invalid file-space page size {p}: must be a power of two >= 512"
                )
            }
            Self::UserblockNotPageAligned(userblock, page_size) => {
                write!(
                    f,
                    "userblock of {userblock} bytes is not a whole number of {page_size}-byte \
                     file-space pages: a paged file measures its pages from the file base, so \
                     the userblock must be a multiple of the page size (or zero)"
                )
            }
            Self::InvalidUserblockSize(size) => {
                write!(
                    f,
                    "invalid userblock size {size}: must be zero or a power of two >= 512"
                )
            }
            Self::UserblockContentTooLarge { content, userblock } => {
                write!(
                    f,
                    "{content} bytes of userblock content do not fit a userblock of {userblock} \
                     bytes"
                )
            }
            Self::InvalidFreeSpaceManager => {
                write!(f, "malformed free-space manager block (FSHD/FSSE)")
            }
            Self::EnumBaseNotInteger => {
                write!(
                    f,
                    "an enumeration's base type must be an integer (fixed-point) type"
                )
            }
            Self::EnumMemberValueSize(name, expected, actual) => {
                write!(
                    f,
                    "enumeration member '{name}' has a {actual}-byte value, but its base type is \
                     {expected} bytes"
                )
            }
            Self::EnumMemberValueRange(name, value, size) => {
                write!(
                    f,
                    "enumeration member '{name}' value {value} does not fit in its \
                     {size}-byte base type"
                )
            }
            Self::InvalidCompoundSize => {
                write!(f, "compound datatype size must be greater than zero")
            }
            Self::EmptyCompoundType => {
                write!(f, "compound datatype must contain at least one field")
            }
            Self::DuplicateCompoundField(name) => {
                write!(f, "duplicate compound field name: {name}")
            }
            Self::CompoundFieldOutOfBounds {
                name,
                offset,
                field_size,
                compound_size,
            } => {
                write!(
                    f,
                    "compound field {name:?} at offset {offset} with size {field_size} \
                     exceeds compound size {compound_size}"
                )
            }
            Self::CompoundFieldOverlap { first, second } => {
                write!(f, "compound fields {first:?} and {second:?} overlap")
            }
            Self::CompoundFieldMissing(name) => {
                write!(f, "compound field {name:?} is missing")
            }
            Self::CompoundFieldTypeMismatch(name) => {
                write!(f, "compound field {name:?} has an incompatible datatype")
            }
            Self::InvalidDataspaceVersion(v) => {
                write!(f, "invalid dataspace version: {v}")
            }
            Self::InvalidDataspaceType(t) => {
                write!(f, "invalid dataspace type: {t}")
            }
            Self::InvalidLayoutVersion(v) => {
                write!(f, "invalid data layout version: {v}")
            }
            Self::InvalidLayoutClass(c) => {
                write!(f, "invalid data layout class: {c}")
            }
            Self::InvalidChunkIndexType(t) => {
                write!(f, "invalid chunk index type: {t}")
            }
            Self::UnreadableFillValue => write!(
                f,
                "the dataset's fill value message could not be parsed, and part of its \
                 storage was never allocated, so those elements have no determined value"
            ),
            Self::TypeMismatch { expected, actual } => {
                write!(f, "type mismatch: expected {expected}, got {actual}")
            }
            Self::DataSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "data size mismatch: expected {expected} bytes, got {actual} bytes"
                )
            }
            Self::InvalidLocalHeapSignature => {
                write!(f, "invalid local heap signature")
            }
            Self::InvalidLocalHeapVersion(v) => {
                write!(f, "invalid local heap version: {v}")
            }
            Self::InvalidLocalHeapName { offset, source } => {
                write!(
                    f,
                    "the local heap name at data segment offset {offset} is not UTF-8: {source}"
                )
            }
            Self::InvalidBTreeSignature => {
                write!(f, "invalid B-tree v1 signature")
            }
            Self::InvalidBTreeNodeType(t) => {
                write!(f, "invalid B-tree node type: {t}")
            }
            Self::InvalidSymbolTableNodeSignature => {
                write!(f, "invalid symbol table node signature")
            }
            Self::InvalidSymbolTableNodeVersion(v) => {
                write!(f, "invalid symbol table node version: {v}")
            }
            Self::PathNotFound(p) => {
                write!(f, "path not found: {p}")
            }
            Self::InvalidLinkVersion(v) => {
                write!(f, "invalid link message version: {v}")
            }
            Self::InvalidLinkType(t) => {
                write!(f, "invalid link type: {t}")
            }
            Self::InvalidLinkInfoVersion(v) => {
                write!(f, "invalid link info message version: {v}")
            }
            Self::InvalidBTreeV2Signature => {
                write!(f, "invalid B-tree v2 signature")
            }
            Self::InvalidBTreeV2Version(v) => {
                write!(f, "invalid B-tree v2 version: {v}")
            }
            Self::InvalidFractalHeapSignature => {
                write!(f, "invalid fractal heap signature")
            }
            Self::InvalidFractalHeapVersion(v) => {
                write!(f, "invalid fractal heap version: {v}")
            }
            Self::InvalidHeapIdType(t) => {
                write!(f, "invalid heap ID type: {t}")
            }
            Self::HugeObjectNotFound(id) => {
                write!(f, "fractal-heap huge object {id} not found in B-tree")
            }
            Self::UnexpectedHugeObjectBTree {
                tree_type,
                record_size,
                required,
            } => {
                write!(
                    f,
                    "fractal-heap huge-objects B-tree is not the expected record type 1: \
                     type {tree_type}, records of {record_size} bytes (type 1 needs {required})"
                )
            }
            Self::UnsupportedFilteredHeapObject => {
                write!(f, "filtered fractal-heap objects are not supported")
            }
            Self::UnsupportedVirtualLayout => {
                write!(f, "virtual (VDS) data layout is not supported")
            }
            Self::UnsupportedExternalStorage => {
                write!(
                    f,
                    "dataset stores its elements in external files (H5Pset_external), \
                     which this reader does not follow"
                )
            }
            Self::InvalidAttributeVersion(v) => {
                write!(f, "invalid attribute message version: {v}")
            }
            Self::InvalidAttributeInfoVersion(v) => {
                write!(f, "invalid attribute info message version: {v}")
            }
            Self::InvalidSharedMessageVersion(v) => {
                write!(f, "invalid shared message version: {v}")
            }
            Self::InvalidAttributeFlags(v) => {
                write!(f, "undefined flag bits in attribute message: {v:#04x}")
            }
            Self::UnsupportedSohmReference => {
                write!(
                    f,
                    "a message references the shared object header message (SOHM) heap, and no readable shared message table was found for it"
                )
            }
            Self::InvalidSohmTableVersion(v) => {
                write!(f, "invalid shared message table version: {v}")
            }
            Self::InvalidSohmIndexCount(n) => {
                write!(f, "invalid shared message index count: {n}")
            }
            Self::InvalidSohmTableSignature => {
                write!(f, "invalid shared message table signature")
            }
            Self::InvalidSohmListSignature => {
                write!(f, "invalid shared message list signature")
            }
            Self::InvalidSohmIndexKind(v) => {
                write!(f, "invalid shared message index storage kind: {v}")
            }
            Self::InvalidSohmBTreeType(v) => {
                write!(
                    f,
                    "a shared message index names a v2 B-tree of type {v}, not a shared message index"
                )
            }
            Self::InvalidSohmRecordLocation(v) => {
                write!(f, "invalid shared message record location: {v}")
            }
            Self::SohmIndexMissing(t) => {
                write!(f, "no shared message index holds messages of type {t:#06x}")
            }
            Self::SharedMessageMissing {
                object_header_address,
                message_type,
            } => {
                write!(
                    f,
                    "object header at {object_header_address} holds no message of type {message_type:#06x} for a shared reference to it"
                )
            }
            Self::UnresolvedSharedMessage(t) => {
                write!(
                    f,
                    "a reference to a shared message of type {t:#06x} cannot be resolved without the file that holds it"
                )
            }
            Self::UnknownCommittedDatatype(path) => {
                write!(f, "no committed datatype is written at path {path:?}")
            }
            Self::CommittedDatatypeMismatch { path, user } => {
                write!(
                    f,
                    "{user} names the committed datatype {path:?} but declares a different type"
                )
            }
            Self::InvalidGlobalHeapSignature => {
                write!(f, "invalid global heap collection signature")
            }
            Self::InvalidGlobalHeapVersion(v) => {
                write!(f, "invalid global heap version: {v}")
            }
            Self::GlobalHeapObjectNotFound {
                collection_address,
                index,
            } => {
                write!(
                    f,
                    "global heap object not found: collection {collection_address:#x}, index {index}"
                )
            }
            Self::VlDataError(msg) => {
                write!(f, "variable-length data error: {msg}")
            }
            Self::VariableLengthElementLimitExceeded { limit, actual } => {
                write!(
                    f,
                    "variable-length element limit exceeded: limit is {limit}, data contains {actual}"
                )
            }
            Self::VariableLengthByteLimitExceeded { limit, required } => {
                write!(
                    f,
                    "variable-length payload limit exceeded: limit is {limit} bytes, \
                     data requires {required} bytes"
                )
            }
            Self::SerializationError(msg) => {
                write!(f, "serialization error: {msg}")
            }
            Self::InvalidLinkName(name) => {
                write!(
                    f,
                    "invalid link name {name:?}: a link name is one component of an object \
                     path, so it is not empty, is not \".\" and holds no \"/\""
                )
            }
            Self::DatasetMissingData => {
                write!(f, "dataset is missing data")
            }
            Self::DatasetMissingShape => {
                write!(f, "dataset is missing shape")
            }
            Self::ShapeDataMismatch {
                expected,
                actual,
                element_size,
            } => {
                write!(
                    f,
                    "shape/data mismatch: shape requires {} elements ({expected} bytes), \
                     but {} elements ({actual} bytes) were supplied",
                    expected / element_size.get(),
                    actual / element_size.get(),
                )
            }
            Self::InvalidChunkGeometry(reason) => {
                write!(f, "invalid chunk geometry: {reason}")
            }
            Self::InvalidFilterPipelineVersion(v) => {
                write!(f, "invalid filter pipeline version: {v}")
            }
            Self::UnsupportedFilter(id) => {
                write!(f, "unsupported filter: {id}")
            }
            Self::FilterError(msg) => {
                write!(f, "filter error: {msg}")
            }
            Self::Fletcher32Mismatch { expected, computed } => {
                write!(
                    f,
                    "fletcher32 mismatch: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            Self::ChunkedReadError(msg) => {
                write!(f, "chunked read error: {msg}")
            }
            Self::ChecksumMismatch { expected, computed } => {
                write!(
                    f,
                    "checksum mismatch: expected {expected:#010x}, computed {computed:#010x}"
                )
            }
            Self::NestingDepthExceeded => {
                write!(f, "maximum nesting/continuation depth exceeded")
            }
            Self::UnsupportedZfp(msg) => {
                write!(f, "unsupported ZFP configuration: {msg}")
            }
            Self::ValueTooLargeForPlatform { value, target } => {
                write!(f, "value {value} does not fit in {target} on this platform")
            }
            Self::OffsetOverflow { offset, length } => {
                write!(
                    f,
                    "offset arithmetic overflow: {offset} + {length} exceeds u64"
                )
            }
            Self::AddressBelowBase { address, base } => {
                write!(
                    f,
                    "file address {address} is below the superblock base address \
                     {base}, so it names no stored (base-relative) position"
                )
            }
            Self::AllocationFailed { values, source } => {
                write!(f, "cannot reserve room for {values} values: {source}")
            }
            Self::Source(msg) => {
                write!(f, "byte source error: {msg}")
            }
            Self::LibverBoundsUnsatisfiable {
                writes,
                requested_low,
                requested_high,
            } => {
                write!(
                    f,
                    "requested library-version bounds [{requested_low}, {requested_high}] \
                     cannot be satisfied: this crate writes the v1.8 and {writes} formats"
                )
            }
            Self::LibverTooOldForContent {
                content,
                needs,
                writing,
            } => {
                write!(
                    f,
                    "{content} requires the {needs} format, but the requested \
                     library-version bounds write {writing}"
                )
            }
            Self::InvalidObjectReference(addr) => {
                write!(
                    f,
                    "invalid HDF5 object reference: address {addr:#x} is null/undefined \
                     or does not point at a group or dataset"
                )
            }
            Self::UnsupportedFillValueVersion(v) => {
                write!(f, "unsupported fill value message version: {v}")
            }
            Self::FillValueSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "fill value size {actual} bytes does not match the dataset datatype \
                     element size of {expected} bytes"
                )
            }
            Self::AttributeMessageTooLarge { name, size } => {
                write!(
                    f,
                    "attribute {name:?} serializes to {size} bytes, past the \
                     {OBJECT_HEADER_MESSAGE_MAX}-byte limit of the object header's \
                     message size field"
                )
            }
            Self::ObjectHeaderMessageTooLarge { message_type, size } => {
                write!(
                    f,
                    "object header message {message_type:#06x} is {size} bytes, past the \
                     {OBJECT_HEADER_MESSAGE_MAX}-byte limit of the object header's \
                     message size field"
                )
            }
            Self::AttributeFieldTooLong {
                name,
                field,
                size,
                limit,
            } => {
                write!(
                    f,
                    "attribute {name:?} has a {size}-byte {field}, past the {limit}-byte limit of \
                     the attribute message's {field} size field"
                )
            }
            Self::DenseAttributeHeapTooLarge { limit } => {
                write!(
                    f,
                    "these attributes need more than the {limit}-byte address space of a dense \
                     attribute heap"
                )
            }
            Self::ZeroFixedStringWidth => {
                write!(
                    f,
                    "a fixed-width string datatype must be at least one byte wide"
                )
            }
            Self::FixedStringTooLong { index, len, width } => {
                write!(
                    f,
                    "string element {index} is {len} bytes, past the declared {width}-byte width"
                )
            }
            Self::NumericElementTooWide { size } => {
                write!(
                    f,
                    "a {size}-byte numeric element is wider than the 64-bit values these readers \
                     decode into"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for FormatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FormatError::AllocationFailed { source, .. } => Some(source),
            FormatError::InvalidLocalHeapName { source, .. } => Some(source),
            _ => None,
        }
    }
}
