use std::fs;

use rstest::rstest;
use test_util::allocation_oracle::Coverage;
use test_util::allocation_oracle::Interval;
use test_util_hdf5::allocation_publication;
use test_util_hdf5::allocation_publication::TailEncoding;

use crate::MemoryStrategy;
use crate::SyncPolicy;

use super::super::Observation;
use super::super::rollback;
use super::super::rollback::Encoding;
use super::super::rollback::WritePath;

#[rstest]
#[case::immediate(WritePath::Immediate)]
#[case::borrowed_write(WritePath::BorrowedWrite)]
#[case::borrowed_flush(WritePath::BorrowedFlush)]
#[case::owned_write(WritePath::OwnedWrite)]
#[case::owned_flush(WritePath::OwnedFlush)]
fn native_append_failures_preserve_deferred_and_existing_indexes(
    #[case] writer: WritePath,
    #[values(false, true)] paged: bool,
    #[values(false, true)] deferred: bool,
    #[values(MemoryStrategy::Mirrored, MemoryStrategy::Bounded)] backing: MemoryStrategy,
    #[values(SyncPolicy::Always, SyncPolicy::OnClose)] policy: SyncPolicy,
    #[values(Encoding::Fletcher32, Encoding::Deflate)] encoding: Encoding,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("native-rollback.h5");
    let native_encoding = match encoding {
        Encoding::Deflate => TailEncoding::Deflate,
        Encoding::Fletcher32 => TailEncoding::Fletcher32,
    };
    allocation_publication::write_tail_fixture(&path, paged, deferred, native_encoding);
    let initial = fs::read(&path).unwrap();
    let before = Observation::read(&initial);
    assert_eq!(before.index_defined, !deferred);
    let native = allocation_publication::Observation::read(&path);
    assert_eq!(native.values, before.values);
    assert_eq!(native.chunks, before.chunks);
    rollback::force_tail(&path);
    rollback::run(&path, writer, deferred, backing, policy, false, encoding);
    let pure = Observation::read(&fs::read(&path).unwrap());
    let native = allocation_publication::Observation::read(&path);
    assert_eq!(native.values, pure.values);
    assert_eq!(native.chunks, pure.chunks);
    assert_eq!(
        Coverage::new(
            native
                .free
                .into_iter()
                .map(|(address, length)| Interval::new(address, length).unwrap())
        ),
        pure.free
    );
}
