#![no_main]

use hdf5_pure::File;
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use tempfile::NamedTempFile;

#[path = "../streaming_differential.rs"]
mod oracle;

fuzz_target!(|data: &[u8]| {
    let buffered = File::from_bytes(data.to_vec());
    let mut temp = match NamedTempFile::new() {
        Ok(temp) => temp,
        Err(_) => return,
    };
    if temp.write_all(data).is_err() || temp.flush().is_err() {
        return;
    }

    let streaming = File::open_streaming(temp.path());

    if let (Ok(buffered), Ok(streaming)) = (buffered, streaming) {
        oracle::assert_same_file_metadata(&buffered, &streaming);
        oracle::assert_same_group(
            &buffered,
            &streaming,
            &buffered.root(),
            &streaming.root(),
            "",
            0,
        );
    }
});

