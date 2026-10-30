#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../streaming_differential.rs"]
mod oracle;

fuzz_target!(|data: &[u8]| {
    oracle::compare(data);
});
