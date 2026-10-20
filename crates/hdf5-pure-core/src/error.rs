use alloc::collections::TryReserveError;
use alloc::string::String;
use core::fmt;
use core::num::NonZeroUsize;
use core::str::Utf8Error;

use crate::FileSpacePageSize;

/// Reports format, filter, and runtime failures through the `hdf5-pure` API.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// An absolute file position lies below the base used for address conversion.
    AddressBelowBase {
        /// The absolute file position that could not be made base-relative.
        address: u64,
        /// The address base used for the conversion.
        base: u64,
    },
    /// An address is too large for an address field of `offset_size` bytes.
    AddressTooLarge {
        /// The address that does not fit the field.
        address: u64,
        /// The address field's width in bytes.
        offset_size: u8,
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
    /// Jenkins lookup3 metadata checksum mismatch.
    ChecksumMismatch {
        /// The checksum stored in the file.
        expected: u32,
        /// The checksum we computed.
        computed: u32,
    },
    /// Chunked dataset read error.
    ChunkedReadError(String),
    /// A dataset or attribute refers to a committed datatype whose encoding differs
    /// from its own.
    CommittedDatatypeMismatch {
        /// Path of the referenced committed datatype object.
        path: String,
        /// Name of the dataset or attribute that named it.
        user: String,
    },
    /// A named compound field was not present.
    CompoundFieldMissing(String),
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
    /// A compound field has an incompatible datatype.
    CompoundFieldTypeMismatch(String),
    /// Data size mismatch.
    DataSizeMismatch {
        /// Expected size in bytes.
        expected: usize,
        /// Actual size in bytes.
        actual: usize,
    },
    /// Dataset is missing data.
    DatasetMissingData,
    /// Dataset is missing shape.
    DatasetMissingShape,
    /// A defined address equals the undefined sentinel value for an address field of `offset_size` bytes.
    DefinedAddressIsUndefined {
        /// The defined address that matches the sentinel value.
        address: u64,
        /// The address field's width in bytes.
        offset_size: u8,
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
    /// A compound datatype contains the same field name more than once.
    DuplicateCompoundField(String),
    /// A global heap collection holds two objects at the same index.
    DuplicateGlobalHeapObjectIndex {
        /// The physical address of the collection.
        collection_address: u64,
        /// The index of the second object.
        index: u16,
    },
    /// A compound datatype contains no fields.
    EmptyCompoundType,
    /// An enumeration datatype was built over a base type that is not an
    /// integer. HDF5 enumerations must have a fixed-point base.
    EnumBaseNotInteger,
    /// An enumeration member's integer value does not fit in the base type:
    /// `(member name, value, base size in bytes)`.
    EnumMemberValueRange(String, i64, u32),
    /// An enumeration member's value does not occupy exactly the base type's
    /// size: `(member name, expected bytes, actual bytes)`.
    EnumMemberValueSize(String, u32, usize),
    /// Relocating the encoded end-of-file address would place it below zero.
    EofRelocationUnderflow {
        /// The end-of-file address encoded in the superblock.
        stored_eof: u64,
        /// The number of bytes by which the HDF5 image moved earlier in the file.
        shift: u64,
    },
    /// The supplied byte count differs from the declared physical range length.
    FileRegionLengthMismatch {
        /// The declared range length in bytes.
        range_length: u64,
        /// The supplied buffer length in bytes.
        bytes_length: u64,
    },
    /// A user-supplied fill value's byte width does not match the dataset's
    /// datatype element size (for example a `u8` fill value on an `i32`
    /// dataset).
    FillValueSizeMismatch {
        /// The dataset datatype's element size in bytes.
        expected: usize,
        /// The supplied fill value's size in bytes.
        actual: usize,
    },
    /// Filter processing failed.
    ///
    /// The string describes the filter and the failure.
    FilterError(String),
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
    /// Fletcher32 checksum mismatch.
    Fletcher32Mismatch {
        /// Expected checksum.
        expected: u32,
        /// Computed checksum.
        computed: u32,
    },
    /// A global heap collection is smaller than 4096 bytes, the minimum collection size.
    GlobalHeapCollectionTooSmall {
        /// The physical address of the collection.
        collection_address: u64,
        /// The collection size its header stores, in bytes.
        size: u64,
    },
    /// Global heap object not found.
    GlobalHeapObjectNotFound {
        /// Address of the collection.
        collection_address: u64,
        /// Index that was not found.
        index: u16,
    },
    /// A global heap object, or the object header of one, runs past the end of its collection.
    GlobalHeapObjectOutsideCollection {
        /// The physical address of the object header.
        address: u64,
        /// The index the object header stores.
        index: u16,
        /// The physical address of the first byte past the collection.
        collection_end: u64,
    },
    /// A fractal-heap "huge" object's heap ID referenced a B-tree key that is
    /// not present in the heap's huge-objects v2 B-tree.
    HugeObjectNotFound(u64),
    /// The supplied physical region does not contain the required range.
    ///
    /// This reports insufficient caller-supplied bytes for a valid range. A short storage read
    /// is reported by the source layer.
    InsufficientRegion {
        /// The required range's absolute start position.
        required_start: u64,
        /// The required range length in bytes.
        required_length: u64,
        /// The supplied region's absolute start position.
        provided_start: u64,
        /// The supplied region length in bytes.
        provided_length: u64,
    },
    /// An internal error, a condition that only a bug in `hdf5-pure` can produce.
    ///
    /// The string describes the condition.
    Internal(String),
    /// An attribute message's flags byte set a bit the format does not define.
    /// Only bit 0 (shared datatype) and bit 1 (shared dataspace) exist, so any
    /// other bit means the message is not what it claims to be.
    InvalidAttributeFlags(u8),
    /// Invalid Attribute Info message version.
    InvalidAttributeInfoVersion(u8),
    /// Invalid attribute message version.
    InvalidAttributeVersion(u8),
    /// Invalid B-tree node type.
    InvalidBTreeNodeType(u8),
    /// Invalid B-tree v1 signature.
    InvalidBTreeSignature,
    /// Invalid B-tree v2 signature.
    InvalidBTreeV2Signature,
    /// Invalid B-tree v2 version.
    InvalidBTreeV2Version(u8),
    /// Invalid byte order.
    InvalidByteOrder(u8),
    /// Invalid character set.
    InvalidCharacterSet(u8),
    /// A chunked, filtered, or extensible dataset's chunk geometry is invalid: for
    /// example chunk dimensions whose rank disagrees with the shape, a zero chunk
    /// dimension, a maximum shape whose rank disagrees with the shape or that is
    /// smaller than the current shape, or chunking requested on a scalar dataset.
    /// The writer rejects malformed geometry before splitting chunks. The payload
    /// is a human-readable reason.
    InvalidChunkGeometry(&'static str),
    /// A chunked layout's chunk indexing type is outside the five the Chunk
    /// Indexing Type table of "The Data Layout Message" defines, in the [format
    /// specification, version 4.0][spec]: Single Chunk (1), Implicit (2), Fixed
    /// Array (3), Extensible Array (4), and version 2 B-tree (5).
    /// `H5O__layout_decode` rejects the same bytes (`H5Olayout.c`, HDF5
    /// 1.14.6).
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_layout
    InvalidChunkIndexType(u8),
    /// A compound datatype has a zero total size.
    InvalidCompoundSize,
    /// Invalid dataspace type.
    InvalidDataspaceType(u8),
    /// Invalid dataspace version.
    InvalidDataspaceVersion(u8),
    /// Invalid datatype class.
    InvalidDatatypeClass(u8),
    /// Invalid datatype version for a given class.
    InvalidDatatypeVersion {
        /// The type class.
        class: u8,
        /// The version found.
        version: u8,
    },
    /// An Extensible Array index block, super block, or data block whose client ID differs from
    /// its header's.
    InvalidExtensibleArrayBlockClient {
        /// The client ID the header stores.
        expected: u8,
        /// The client ID the block stores.
        actual: u8,
    },
    /// An Extensible Array super block or data block whose block offset differs from its position
    /// in the array.
    InvalidExtensibleArrayBlockOffset {
        /// The block's position in the array, the number of elements before it that the index
        /// block does not hold.
        expected: u64,
        /// The block offset the block stores.
        actual: u64,
    },
    /// An Extensible Array index block, super block, or data block whose header address differs
    /// from the address of the array's header.
    InvalidExtensibleArrayBlockOwner {
        /// The address of the header.
        expected: u64,
        /// The header address the block stores, in the coordinates of `expected`.
        actual: u64,
    },
    /// A reserved client ID in an Extensible Array header, 2 or more.
    InvalidExtensibleArrayClient(u8),
    /// An Extensible Array header's element size or creation parameters are out of range, or a
    /// length or a slot the parser derives from them is past `u64::MAX`.
    ///
    /// The parser returns it for a header with:
    ///
    /// - a filtered element size shorter than an address and a filter mask, or with a chunk size
    ///   wider than 8 bytes
    /// - a maximum element bit count outside `1..=64`
    /// - a minimum data block element count that is not a power of two
    /// - a minimum data block pointer count of a super block that is not a power of two, or is 1
    /// - a page bit count past the maximum element bit count or past 63
    /// - creation parameters whose blocks do not fit the maximum element count, or that page a
    ///   data block whose address the index block holds
    InvalidExtensibleArrayGeometry,
    /// An Extensible Array header that does not start with `EAHD`, or an index block, super block,
    /// or data block that does not start with `EAIB`, `EASB`, or `EADB`.
    InvalidExtensibleArraySignature,
    /// An Extensible Array header or block version other than 0, the only version the format
    /// defines.
    InvalidExtensibleArrayVersion(u8),
    /// A file space page size is outside 512 bytes to 1 GiB, the range of [`FileSpacePageSize`].
    InvalidFileSpacePageSize(u64),
    /// Invalid file-space management strategy code in a File Space Info message.
    InvalidFileSpaceStrategy(u8),
    /// A fill value write time code that the format does not define.
    InvalidFillTime(u8),
    /// A fill value defined indicator byte that the format does not define.
    InvalidFillValueDefined(u8),
    /// A version 3 fill value flags byte with reserved bits set or conflicting defined flags.
    InvalidFillValueFlags(u8),
    /// An object header message type that is not a fill value message type.
    InvalidFillValueMessageType(u16),
    /// A stored filter name violates the pipeline's name encoding.
    InvalidFilterName {
        /// The identifier of the filter with the malformed name.
        filter_id: u16,
        /// The name constraint that failed.
        reason: &'static str,
    },
    /// A filter pipeline field exceeds its format limit.
    InvalidFilterPipelineField {
        /// The field whose value exceeds the limit.
        field: &'static str,
        /// The value read from the field.
        value: usize,
        /// The largest value the format permits for the field.
        maximum: usize,
    },
    /// Invalid filter pipeline version.
    InvalidFilterPipelineVersion(u8),
    /// A Fixed Array data block's client ID differs from its header's.
    InvalidFixedArrayBlockClient {
        /// The client ID the header stores.
        expected: u8,
        /// The client ID the data block stores.
        actual: u8,
    },
    /// A Fixed Array data block's header address differs from the address of the header that
    /// points at it.
    InvalidFixedArrayBlockOwner {
        /// The address of the header.
        expected: u64,
        /// The header address the data block stores, in the coordinates of `expected`.
        actual: u64,
    },
    /// A reserved client ID in a Fixed Array header, 2 or more.
    InvalidFixedArrayClient(u8),
    /// A Fixed Array header's page bits, entry size, or entry count is out of range.
    ///
    /// The parser returns it for page bits of 64 or more, for a filtered entry size shorter than
    /// an address and a filter mask or with a chunk size wider than 8 bytes, and for an entry
    /// count that makes the data block longer than `u64::MAX` bytes.
    InvalidFixedArrayGeometry,
    /// A Fixed Array header that does not start with `FAHD`, or a data block that does not start
    /// with `FADB`.
    InvalidFixedArraySignature,
    /// A Fixed Array header or data block version other than 0, the only version the format
    /// defines.
    InvalidFixedArrayVersion(u8),
    /// A fractal-heap block's stored offset differs from its position in the doubling table.
    InvalidFractalHeapBlockOffset {
        /// The block's expected offset in the heap's managed address space, in bytes.
        expected: u64,
        /// The block offset read from its prefix, in bytes.
        actual: u64,
    },
    /// A fractal-heap block's header address differs from the expected heap owner.
    InvalidFractalHeapBlockOwner {
        /// The owning heap header's base-relative address.
        expected: u64,
        /// The heap header's base-relative address read from the block prefix.
        actual: u64,
    },
    /// A fractal heap's doubling table, block extent, or filter metadata is invalid.
    InvalidFractalHeapGeometry,
    /// A supplied heap ID has a different length from the heap header's declared ID length.
    InvalidFractalHeapIdLength {
        /// The declared heap ID length in bytes.
        expected: usize,
        /// The supplied heap ID length in bytes.
        actual: usize,
    },
    /// Invalid fractal heap signature.
    InvalidFractalHeapSignature,
    /// Invalid fractal heap version.
    InvalidFractalHeapVersion(u8),
    /// A free-space manager block (`FSHD` or `FSSE`) is malformed or past the end of the file, or a
    /// free section is too large for a manager at the file's length width.
    ///
    /// The string describes the condition.
    InvalidFreeSpaceManager(String),
    /// Invalid global heap collection signature.
    InvalidGlobalHeapSignature,
    /// Invalid global heap version.
    InvalidGlobalHeapVersion(u8),
    /// A version 0 or 1 superblock stores 0 in its "Group Leaf Node K" field. The field is
    /// defined in "Format Signature and Superblock" of the [format specification, version
    /// 4.0][spec], which requires a value greater than zero.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_boot_super
    InvalidGroupLeafNodeK,
    /// Invalid heap ID type.
    InvalidHeapIdType(u8),
    /// Invalid data layout class.
    InvalidLayoutClass(u8),
    /// Invalid data layout version.
    InvalidLayoutVersion(u8),
    /// Invalid length size (must be 2, 4, or 8).
    InvalidLengthSize(u8),
    /// Invalid Link Info message version.
    InvalidLinkInfoVersion(u8),
    /// A name a builder was given for a dataset, a group, or a committed datatype is not a
    /// component of an object path: the empty name, `.`, or a name holding `/`. The object
    /// under such a link has no path, and the write reports the name it rejects.
    InvalidLinkName(String),
    /// Invalid link type code.
    InvalidLinkType(u8),
    /// Invalid Link message version.
    InvalidLinkVersion(u8),
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
    /// A link name offset at or past the end of the data segment of a group's local heap.
    InvalidLocalHeapNameOffset {
        /// The name's offset within the heap's data segment.
        offset: u64,
        /// The size of the data segment the heap header stores, in bytes.
        data_segment_size: u64,
    },
    /// Invalid local heap signature.
    InvalidLocalHeapSignature,
    /// Invalid local heap version.
    InvalidLocalHeapVersion(u8),
    /// A version 1 object-header message has a data size that is not a multiple of eight.
    ///
    /// Contains the declared message data size.
    InvalidObjectHeaderMessageSize(u16),
    /// Invalid object header signature.
    InvalidObjectHeaderSignature,
    /// Invalid object header version.
    InvalidObjectHeaderVersion(u8),
    /// An HDF5 object reference (`H5R_OBJECT`) could not be resolved to an
    /// object: the stored address is null or undefined (`HADDR_UNDEF`), or it
    /// does not point at a group or dataset object header. The payload is the
    /// stored (base-relative) address, preserved for diagnostics.
    InvalidObjectReference(u64),
    /// Invalid offset size (must be 2, 4, or 8).
    InvalidOffsetSize(u8),
    /// Invalid reference type.
    InvalidReferenceType(u8),
    /// Invalid shared message version.
    InvalidSharedMessageVersion(u8),
    /// A shared-message index refers to a version 2 B-tree with a type other
    /// than shared-message index (type 7). Contains the B-tree type.
    InvalidSohmBTreeType(u8),
    /// A Shared Message Table message declares no indexes, or more than the
    /// eight the format allows. The count sizes the table's own read, so a
    /// wrong one reads neighbouring bytes as index headers.
    InvalidSohmIndexCount(u8),
    /// A shared-message index header declares a storage kind that is neither a
    /// list (0) nor a version 2 B-tree (1).
    InvalidSohmIndexKind(u8),
    /// A shared-message list index does not start with its `SMLI` signature.
    InvalidSohmListSignature,
    /// A shared-message index record identifies a storage location that is neither
    /// the heap (0) nor an object header (1).
    InvalidSohmRecordLocation(u8),
    /// The shared-message table does not start with its `SMTB` signature.
    InvalidSohmTableSignature,
    /// A file's shared-message table, or a Shared Message Table message identifying
    /// it, declares a version this format does not define.
    InvalidSohmTableVersion(u8),
    /// A space allocation time code that the format does not define.
    InvalidSpaceAllocationTime(u8),
    /// Invalid string padding type.
    InvalidStringPadding(u8),
    /// Invalid symbol table node signature.
    InvalidSymbolTableNodeSignature,
    /// A symbol table node stores more symbols than the 2K entries of a node, where K is the
    /// superblock's "Group Leaf Node K". The node is defined in "Group Symbol Table Nodes" of the
    /// [format specification, version 4.0][spec].
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_symboltable
    InvalidSymbolTableNodeSymbolCount {
        /// The number of symbols the node stores.
        symbols: u16,
        /// The number of entries in a node, twice the superblock's "Group Leaf Node K".
        capacity: u32,
    },
    /// Invalid symbol table node version.
    InvalidSymbolTableNodeVersion(u8),
    /// A userblock size the format does not define: it must be zero, or a power
    /// of two of at least 512 bytes. A reader looks for the superblock at 0, 512,
    /// 1024, and so on doubling, so any other size produces a file nothing can
    /// open.
    InvalidUserblockSize(u64),
    /// A length is too large for a length field of `length_size` bytes.
    LengthTooLarge {
        /// The length that does not fit the field.
        length: u64,
        /// The length field's width in bytes.
        length_size: u8,
    },
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
    /// The free space object of a global heap collection does not span the rest of the
    /// collection.
    MisplacedGlobalHeapFreeSpace {
        /// The physical address of the free space object's header.
        address: u64,
        /// The size the free space object stores, its header included, in bytes.
        size: u64,
        /// The physical address of the first byte past the collection.
        collection_end: u64,
    },
    /// Maximum nesting/continuation depth exceeded (malformed data protection).
    NestingDepthExceeded,
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
    /// Address or size arithmetic exceeded the range of its integer type.
    OffsetOverflow {
        /// First operand (typically the base offset/address).
        offset: u64,
        /// Second operand (typically the length/size).
        length: u64,
    },
    /// Path not found during group traversal.
    PathNotFound(String),
    /// Serialization error.
    SerializationError(String),
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
    /// A shared message reference identifies an object header without a message
    /// of the referenced type.
    SharedMessageMissing {
        /// Address of the object header identified by the reference.
        object_header_address: u64,
        /// Raw type ID of the message the reference stood in for.
        message_type: u16,
    },
    /// The HDF5 magic signature was not found at any valid offset.
    SignatureNotFound,
    /// A message body references the shared-message heap without a matching
    /// index that covers its type and has an allocated heap. Contains the raw
    /// type ID of the referenced message.
    SohmIndexMissing(u16),
    /// A random-access byte source failed to supply bytes.
    ///
    /// The string describes the backend-specific failure.
    Source(String),
    /// The physical file is shorter than its recorded end-of-file position after relocation.
    TruncatedFile {
        /// The physical end-of-file position derived from the superblock and signature location.
        expected_eof: u64,
        /// The physical backing-store length in bytes.
        actual_len: u64,
    },
    /// Type mismatch when reading data.
    TypeMismatch {
        /// Expected type description.
        expected: &'static str,
        /// Actual type description.
        actual: &'static str,
    },
    /// The size of a global heap collection is not a multiple of 8 bytes.
    UnalignedGlobalHeapCollection {
        /// The physical address of the collection.
        collection_address: u64,
        /// The collection size its header stores, in bytes.
        size: u64,
    },
    /// An Object Header Continuation message stores the undefined address, all bits set, as the
    /// address of its continuation block.
    UndefinedContinuationAddress,
    /// A local heap header stores the undefined address, all bits set, as the address of its data
    /// segment.
    UndefinedLocalHeapDataSegment,
    /// Unexpected end of data.
    UnexpectedEof {
        /// Number of bytes expected.
        expected: usize,
        /// Number of bytes actually available.
        available: usize,
    },
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
    /// A dataset or attribute refers to a committed datatype at a path where the file
    /// being written places no such object.
    UnknownCommittedDatatype(String),
    /// The dataset's Fill Value message could not be parsed, and a read needed
    /// it: part of the dataset's storage was never allocated, so what those
    /// elements read as is undetermined. A dataset whose storage is fully
    /// allocated reads normally regardless of this message.
    UnreadableFillValue,
    /// A shared-message reference could not be resolved without access to its source file.
    ///
    /// Contains the raw type ID of the referenced message.
    UnresolvedSharedMessage(u16),
    /// A dataset's element bytes live in files outside this one
    /// (`H5Pset_external`, the External Data Files header message). Its layout
    /// message is contiguous with an undefined data address, the same encoding
    /// used by an unwritten dataset. The reader rejects external storage because
    /// it cannot follow the external files.
    UnsupportedExternalStorage,
    /// Unsupported File Space Info message version: the parser reads versions 0 and 1.
    UnsupportedFileSpaceInfoVersion(u8),
    /// A Fill Value message (`0x0005`) has an on-disk version this crate does
    /// not recognize (only 1, 2, and 3 are defined). The payload is the version
    /// byte found.
    UnsupportedFillValueVersion(u8),
    /// Unsupported filter ID.
    UnsupportedFilter(u16),
    /// A filtered fractal heap holds an object or structure that this crate does not decode or
    /// encode, such as a huge object.
    UnsupportedFilteredHeapObject,
    /// Unknown message type that is marked as must-understand.
    UnsupportedMessage(u16),
    /// A serialized free-space section class is unsupported for its manager client.
    ///
    /// Fractal-heap classes 2 and 3 are defined in "Free-space Index" of the [format
    /// specification, version 4.0][spec]. Parsing supports their reconstructed geometry.
    /// Serialized records of these classes and reserved classes report this error.
    ///
    /// [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsec_fmt4_infra_freespaceindex
    UnsupportedSerializedSectionClass {
        /// The client identifier read from the free-space manager header.
        client_id: u8,
        /// The unsupported class identifier read from the serialized section record.
        class_id: u8,
    },
    /// A message body references the shared object header message (SOHM) heap,
    /// but parsing lacks a shared-message table. The superblock extension may
    /// omit the table, the table may be unreadable, or the caller may parse a
    /// message body without its file. A committed (`H5Tcommit`) datatype
    /// references another object header and is resolved separately.
    UnsupportedSohmReference,
    /// The superblock version is not supported.
    UnsupportedVersion(u8),
    /// A dataset uses the Virtual (VDS) data layout, which maps its elements to
    /// regions of other datasets, possibly in other files. The reader rejects a
    /// dataset whose virtual mappings it cannot resolve.
    UnsupportedVirtualLayout,
    /// ZFP filter configuration is invalid (e.g. missing element type, rank out of range).
    UnsupportedZfp(String),
    /// A link name in a group's local heap has no NUL terminator before the end of the data
    /// segment.
    UnterminatedLocalHeapName {
        /// The name's offset within the heap's data segment.
        offset: u64,
    },
    /// More userblock content was supplied than the userblock region holds. The
    /// overflow would displace the superblock, so the writer rejects the content.
    UserblockContentTooLarge {
        /// Bytes supplied.
        content: u64,
        /// Bytes the userblock region holds.
        userblock: u64,
    },
    /// A paged file-space strategy was requested alongside a userblock that is
    /// not a whole number of pages. File-space pages are measured from the file
    /// base, so the two boundaries coincide only when the userblock divides by
    /// the page size: `(userblock bytes, page size)`.
    UserblockNotPageAligned(u64, u64),
    /// An integer exceeds the range of a host-sized target type.
    ///
    /// On a 32-bit host, a conversion to [`usize`] reports this error for values above
    /// [`u32::MAX`].
    ValueTooLargeForPlatform {
        /// The original value before conversion.
        value: u64,
        /// The host-sized target type, such as `"usize"`.
        target: &'static str,
    },
    /// An integer exceeds the range of a fixed-width target type.
    ///
    /// The limit is independent of the host pointer width.
    ValueTooLargeForRepresentation {
        /// The original value before conversion.
        value: u64,
        /// The fixed-width target type, such as `"u32"`.
        target: &'static str,
    },
    /// A variable-length read exceeded its configured payload-byte limit.
    VariableLengthByteLimitExceeded {
        /// Maximum number of payload bytes permitted by the caller.
        limit: usize,
        /// Number of payload bytes required by the selected data.
        required: u64,
    },
    /// A variable-length read exceeded its configured element limit.
    VariableLengthElementLimitExceeded {
        /// Maximum number of elements permitted by the caller.
        limit: usize,
        /// Number of elements present in the selected data.
        actual: u64,
    },
    /// Variable-length data error.
    VlDataError(String),
    /// A caller passed zero as the declared width of a fixed-width string.
    /// HDF5 string datatypes require a positive width. Explicitly sized dataset
    /// and attribute strings can report this error. A width derived from empty
    /// values is one byte.
    ZeroFixedStringWidth,
    /// A parsed datatype occupies zero bytes per element.
    ZeroSizedDatatype {
        /// The type class that declared it.
        class: u8,
    },
}

pub const OBJECT_HEADER_MESSAGE_MAX: usize = u16::MAX as usize;

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileRegionLengthMismatch {
                range_length,
                bytes_length,
            } => write!(
                f,
                "file region length is {range_length} bytes, supplied buffer has {bytes_length} bytes"
            ),
            Self::InsufficientRegion {
                required_start,
                required_length,
                provided_start,
                provided_length,
            } => write!(
                f,
                "required file range at {required_start} with length {required_length} is outside \
                 supplied region at {provided_start} with length {provided_length}"
            ),
            Self::SignatureNotFound => {
                write!(f, "HDF5 signature not found at any valid offset")
            }
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported superblock version: {v}")
            }
            Self::TruncatedFile {
                expected_eof,
                actual_len,
            } => write!(
                f,
                "truncated file: recorded EOF is {expected_eof}, physical length is {actual_len}"
            ),
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
            Self::AddressTooLarge {
                address,
                offset_size,
            } => {
                write!(
                    f,
                    "address {address:#x} does not fit a {offset_size}-byte address field"
                )
            }
            Self::DefinedAddressIsUndefined {
                address,
                offset_size,
            } => {
                write!(
                    f,
                    "defined address {address:#x} equals the undefined sentinel for a {offset_size}-byte address field"
                )
            }
            Self::LengthTooLarge {
                length,
                length_size,
            } => {
                write!(
                    f,
                    "length {length} does not fit a {length_size}-byte length field"
                )
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
            Self::UndefinedContinuationAddress => {
                write!(
                    f,
                    "object header continuation message stores the undefined address"
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
                    "invalid file-space page size {p}: must be from {} to {} bytes",
                    FileSpacePageSize::MIN.get(),
                    FileSpacePageSize::MAX.get(),
                )
            }
            Self::InvalidFillTime(t) => {
                write!(f, "invalid fill value write time: {t}")
            }
            Self::InvalidFillValueDefined(d) => {
                write!(f, "invalid fill value defined indicator: {d}")
            }
            Self::InvalidFillValueFlags(flags) => {
                write!(f, "invalid fill value flags: {flags:#04x}")
            }
            Self::InvalidFillValueMessageType(t) => {
                write!(f, "invalid message type for fill value: {t:#06x}")
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
            Self::UnsupportedSerializedSectionClass {
                client_id,
                class_id,
            } => {
                write!(
                    f,
                    "unsupported serialized free-space section class {class_id} for client {client_id}"
                )
            }
            Self::InvalidFreeSpaceManager(reason) => {
                write!(f, "invalid free-space manager: {reason}")
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
            Self::InvalidLocalHeapNameOffset {
                offset,
                data_segment_size,
            } => {
                write!(
                    f,
                    "local heap name offset {offset} lies outside the {data_segment_size}-byte data segment"
                )
            }
            Self::UnterminatedLocalHeapName { offset } => {
                write!(
                    f,
                    "the local heap name at data segment offset {offset} has no NUL terminator in the data segment"
                )
            }
            Self::UndefinedLocalHeapDataSegment => {
                write!(
                    f,
                    "local heap header stores the undefined address as its data segment address"
                )
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
            Self::InvalidGroupLeafNodeK => {
                write!(
                    f,
                    "superblock stores a Group Leaf Node K of 0, and the value must be greater than zero"
                )
            }
            Self::InvalidSymbolTableNodeSymbolCount { symbols, capacity } => {
                write!(
                    f,
                    "symbol table node holds {symbols} symbols, more than its capacity of {capacity}"
                )
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
            Self::InvalidExtensibleArrayBlockClient { expected, actual } => {
                write!(
                    f,
                    "invalid extensible array block client ID: expected {expected}, got {actual}"
                )
            }
            Self::InvalidExtensibleArrayBlockOffset { expected, actual } => {
                write!(
                    f,
                    "invalid extensible array block offset: expected {expected}, got {actual}"
                )
            }
            Self::InvalidExtensibleArrayBlockOwner { expected, actual } => {
                write!(
                    f,
                    "invalid extensible array block header address: expected {expected}, got \
                     {actual}"
                )
            }
            Self::InvalidExtensibleArrayClient(id) => {
                write!(f, "invalid extensible array client ID: {id}")
            }
            Self::InvalidExtensibleArrayGeometry => {
                write!(f, "invalid extensible array geometry")
            }
            Self::InvalidExtensibleArraySignature => {
                write!(f, "invalid extensible array signature")
            }
            Self::InvalidExtensibleArrayVersion(v) => {
                write!(f, "invalid extensible array version: {v}")
            }
            Self::InvalidFixedArrayBlockClient { expected, actual } => {
                write!(
                    f,
                    "invalid fixed array data block client ID: expected {expected}, got {actual}"
                )
            }
            Self::InvalidFixedArrayBlockOwner { expected, actual } => {
                write!(
                    f,
                    "invalid fixed array data block header address: expected {expected}, got \
                     {actual}"
                )
            }
            Self::InvalidFixedArrayClient(id) => {
                write!(f, "invalid fixed array client ID: {id}")
            }
            Self::InvalidFixedArrayGeometry => {
                write!(f, "invalid fixed array geometry")
            }
            Self::InvalidFixedArraySignature => {
                write!(f, "invalid fixed array signature")
            }
            Self::InvalidFixedArrayVersion(v) => {
                write!(f, "invalid fixed array version: {v}")
            }
            Self::InvalidFractalHeapBlockOffset { expected, actual } => {
                write!(
                    f,
                    "invalid fractal heap block offset: expected {expected}, got {actual}"
                )
            }
            Self::InvalidFractalHeapBlockOwner { expected, actual } => {
                write!(
                    f,
                    "invalid fractal heap block owner: expected {expected}, got {actual}"
                )
            }
            Self::InvalidFractalHeapGeometry => {
                write!(f, "invalid fractal heap geometry")
            }
            Self::InvalidFractalHeapIdLength { expected, actual } => {
                write!(
                    f,
                    "invalid fractal heap ID length: expected {expected} bytes, got {actual}"
                )
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
            Self::InvalidSpaceAllocationTime(t) => {
                write!(f, "invalid space allocation time: {t}")
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
            Self::DuplicateGlobalHeapObjectIndex {
                collection_address,
                index,
            } => {
                write!(
                    f,
                    "global heap collection {collection_address:#x} holds index {index} more than once"
                )
            }
            Self::GlobalHeapCollectionTooSmall {
                collection_address,
                size,
            } => {
                write!(
                    f,
                    "global heap collection {collection_address:#x} is {size} bytes, below the \
                     minimum"
                )
            }
            Self::UnalignedGlobalHeapCollection {
                collection_address,
                size,
            } => {
                write!(
                    f,
                    "global heap collection {collection_address:#x} is {size} bytes, which \
                     breaks the collection's alignment"
                )
            }
            Self::GlobalHeapObjectOutsideCollection {
                address,
                index,
                collection_end,
            } => {
                write!(
                    f,
                    "global heap object {index} at {address:#x} runs past the collection end \
                     {collection_end:#x}"
                )
            }
            Self::MisplacedGlobalHeapFreeSpace {
                address,
                size,
                collection_end,
            } => {
                write!(
                    f,
                    "global heap free space at {address:#x} spans {size} bytes and does not end \
                     at the collection end {collection_end:#x}"
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
            Self::InvalidFilterName { filter_id, reason } => {
                write!(f, "invalid name for filter {filter_id}: {reason:?}")
            }
            Self::InvalidFilterPipelineField {
                field,
                value,
                maximum,
            } => {
                write!(
                    f,
                    "filter pipeline {field:?} is {value}, above maximum {maximum}"
                )
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
            Self::ValueTooLargeForRepresentation { value, target } => {
                write!(f, "value {value} does not fit in {target:?}")
            }
            Self::EofRelocationUnderflow { stored_eof, shift } => {
                write!(
                    f,
                    "EOF relocation underflow: {stored_eof} - {shift} is below zero"
                )
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
            Self::Internal(detail) => write!(f, "internal error in hdf5-pure: {detail}"),
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
