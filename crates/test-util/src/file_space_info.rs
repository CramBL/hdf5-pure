//! File Space Info message bodies, and their replacement in a file.
//!
//! The message is defined in "The File Space Info Message" of the [format specification, version
//! 4.0][spec].
//!
//! [spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#subsubsec_fmt4_dataobject_hdr_msg_fsinfo

use std::path::Path;

use crate::bytes;
use crate::object_header::MessageType;
use crate::object_header::v2;
use crate::widths::Widths;

/// Replaces the body of the File Space Info message in the file at `path` with `body`.
///
/// `extension` is the absolute file position of the superblock extension's object header, which
/// holds the message.
///
/// # Panics
///
/// Panics as [`v2::replace_message`] does, and if the file cannot be read or written.
pub fn replace_message(path: impl AsRef<Path>, extension: u64, body: &[u8]) {
    let path = path.as_ref();
    let mut file = crate::read_file(path);
    v2::replace_message(
        &mut file,
        usize::try_from(extension).unwrap(),
        MessageType::FILE_SPACE_INFO,
        body,
    );
    std::fs::write(path, &file).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
}

/// Returns the body of a version 0 message: the version, `strategy`, `threshold` at the length
/// width of `widths`, and each of `manager_addrs` at its offset width.
pub fn version_0(widths: Widths, strategy: u8, threshold: u64, manager_addrs: &[u64]) -> Vec<u8> {
    let mut body = vec![VERSION_0, strategy];
    bytes::push_uint(&mut body, threshold, widths.length);
    for &addr in manager_addrs {
        bytes::push_uint(&mut body, addr, widths.offset);
    }
    body
}

/// Returns the body of a version 1 message that does not persist free space: the version,
/// `strategy`, a clear persist flag, `threshold` and `page_size` at the length width of `widths`,
/// a page-end metadata threshold of 0, and the undefined end of allocation at the offset width.
pub fn non_persistent_version_1(
    widths: Widths,
    strategy: u8,
    threshold: u64,
    page_size: u64,
) -> Vec<u8> {
    let mut body = vec![VERSION_1, strategy, 0];
    bytes::push_uint(&mut body, threshold, widths.length);
    bytes::push_uint(&mut body, page_size, widths.length);
    body.extend_from_slice(&[0, 0]);
    bytes::push_address(&mut body, None, widths.offset);
    body
}

/// The version 1 strategy code for paged allocation, `H5F_FSPACE_STRATEGY_PAGE`.
pub const PAGE: u8 = 1;

/// The version 0 strategy code for free-space managers that persist free space,
/// `H5F_FILE_SPACE_ALL_PERSIST`.
pub const ALL_PERSIST: u8 = 1;

/// The version 0 strategy code for free-space managers that do not persist free space,
/// `H5F_FILE_SPACE_ALL`.
pub const ALL: u8 = 2;

/// The version 0 strategy code for aggregators and the file driver, `H5F_FILE_SPACE_AGGR_VFD`.
pub const AGGR_VFD: u8 = 3;

/// The version 0 strategy code for the file driver alone, `H5F_FILE_SPACE_VFD`.
pub const VFD: u8 = 4;

/// The number of free-space manager addresses a persisting version 0 message stores.
pub const VERSION_0_MANAGERS: usize = 6;

const VERSION_0: u8 = 0;

const VERSION_1: u8 = 1;
