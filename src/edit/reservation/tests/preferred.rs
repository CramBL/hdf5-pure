use core::cell::Cell;
use std::io::ErrorKind;

use hdf5_pure_core::__private::ByteLength;
use hdf5_pure_core::PhysicalAddress;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use hdf5_pure_space::__private::ReservationWork;
use hdf5_pure_space::__private::{ManagerKind, PageType, SessionSpace};
use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;

use crate::FileSpacePageSize;
use crate::FileSpaceStrategy;
use crate::address::{
    AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
};
use crate::edit::block::{BlockRequest, EncodingDependency};
use crate::edit::reservation::{ReservationChoice, ReservationFailure};
use crate::error::{Error, FormatError};
use crate::image::WriteBuffering;
use crate::width::OffsetWidth;

#[rstest]
#[case::exact(0)]
#[case::sentinel(1)]
#[case::beyond(2)]
fn preferred_reuse_checks_complete_narrow_capacity_and_retains_ownership(
    #[case] beyond: u64,
    #[values(OffsetWidth::Two, OffsetWidth::Four)] offsets: OffsetWidth,
    #[values(0, 512)] base: u64,
) {
    let undefined = StoredAddress::undefined(offsets).get();
    let end = undefined - 1 + beyond;
    let stored = end - base - 64;
    let addresses = AddressContext::from_effective_base(BaseAddress::new(base));
    let request = BlockRequest::new(
        super::request(64, PageType::Raw),
        16,
        EncodingDependency::Independent,
    )
    .unwrap();
    assert!(base + stored + request.encoded_length() < undefined);
    let mut space = super::flat(&[(stored, 64)]);
    let mut batch = space.reservation_batch(1);
    let reservation = batch.reserve(request.reservation()).unwrap();
    batch.finish();
    let mut unresolved = None;
    let result =
        super::ReservationPlan::reused(addresses, offsets, reservation, request, &mut unresolved);
    if beyond == 0 {
        let planned = result.unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!(planned.range().start().get(), base + stored);
        assert_eq!(planned.range().end().get(), end);
        assert_eq!(planned.block.encoding().length(), 16);
        assert!(unresolved.is_none());
    } else {
        let Err(Error::Format(error)) = result else {
            panic!("expected a narrow-capacity format error")
        };
        assert_eq!(
            error,
            if beyond == 1 {
                FormatError::DefinedAddressIsUndefined {
                    address: end,
                    offset_size: offsets.get(),
                }
            } else {
                FormatError::AddressTooLarge {
                    address: end,
                    offset_size: offsets.get(),
                }
            }
        );
        assert_eq!(
            unresolved.as_ref().map(|reservation| reservation.extent()),
            Some(super::extent(stored, 64))
        );
        let planned = super::ReservationPlan::reused(
            addresses,
            OffsetWidth::Eight,
            unresolved.take().unwrap(),
            request,
            &mut unresolved,
        )
        .unwrap_or_else(|error| panic!("{error:?}"));
        assert_eq!(planned.range().start().get(), base + stored);
        assert_eq!(planned.range().end().get(), end);
        assert!(unresolved.is_none());
    }
    assert_eq!(space.reusable_sections().collect::<Vec<_>>(), []);
}

#[rstest]
#[case::preferred(64, ReservationChoice::Preferred, &[0, 16, 48][..], 0)]
#[case::fragmented(48, ReservationChoice::Fallback, &[0, 16, 128][..], 3)]
fn preferred_reuse_preserves_ties_and_builds_only_the_needed_batch(
    #[case] length: u64,
    #[case] expected_choice: ReservationChoice,
    #[case] offsets: &[u64],
    #[case] indexed: u64,
    #[values(FileSpaceStrategy::FsmAggr, FileSpaceStrategy::Page)] strategy: FileSpaceStrategy,
    #[values(0, 4096)] base: u64,
) {
    let (_directory, mut engine) = super::engine(strategy, base);
    engine.aggregation = None;
    let start = engine.append(&[13; 512]).unwrap();
    let end = engine.image.len();
    let stored = start - base;
    engine.space = SessionSpace::from_persisted(
        Some(strategy),
        ByteLength::new(1),
        FileSpacePageSize::DEFAULT,
        [0, 128, 256].map(|offset| {
            (
                Some(ManagerKind::PagedSmallRaw),
                super::extent(stored + offset, length),
            )
        }),
    )
    .unwrap();
    let calls = Cell::new(0);
    let (choice, mut plan) = engine
        .plan_blocks_with_fallback(block(64), &[], || {
            calls.set(calls.get() + 1);
            Ok(vec![block(16), block(32), block(16)])
        })
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    assert_eq!(choice, expected_choice);
    assert_eq!(calls.get(), u8::from(choice == ReservationChoice::Fallback));
    assert_eq!(plan.end(), PhysicalAddress::new(end));
    assert_eq!(plan.work().indexed_extents, indexed);
    match choice {
        ReservationChoice::Preferred => {
            assert_eq!(plan.reservations().len(), 1);
            assert_eq!(plan.reservations()[0].range().start().get(), start);
            plan.write_next_with(|writer| {
                writer.write(&[17; 16])?;
                writer.write(&[19; 32])?;
                writer.write(&[23; 16])
            })
            .unwrap();
        }
        ReservationChoice::Fallback => {
            assert_eq!(
                plan.reservations()
                    .iter()
                    .map(|entry| entry.range().start().get())
                    .collect::<Vec<_>>(),
                offsets
                    .iter()
                    .map(|offset| start + offset)
                    .collect::<Vec<_>>()
            );
            plan.write_next(&[17; 16]).unwrap();
            plan.write_next(&[19; 32]).unwrap();
            plan.write_next(&[23; 16]).unwrap();
        }
    }
    let work = plan.finish().unwrap();
    assert_eq!(work.indexed_extents, indexed);
    assert_eq!(work.restored_extents, if indexed == 0 { 0 } else { 2 });
    assert_eq!(engine.image.len(), end);
    for (offset, bytes) in offsets
        .iter()
        .zip([&[17; 16][..], &[19; 32][..], &[23; 16][..]])
    {
        assert_eq!(
            engine
                .image()
                .read_exact_at(start + offset, bytes.len())
                .unwrap(),
            bytes
        );
    }
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        if indexed == 0 {
            vec![
                super::extent(stored + 128, length),
                super::extent(stored + 256, length),
            ]
        } else {
            vec![
                super::extent(stored + 144, 32),
                super::extent(stored + 256, length),
            ]
        }
    );
}

#[rstest]
#[case::bridge(false, false)]
#[case::suffix(true, false)]
#[case::fallback(false, true)]
fn preferred_failures_retain_checked_and_unresolved_ownership(
    #[case] suffix: bool,
    #[case] fallback: bool,
) {
    let (_directory, mut engine) = super::engine(FileSpaceStrategy::FsmAggr, 512);
    engine.aggregation = None;
    let start = engine.append(&[13; 64]).unwrap();
    let end = engine.image.len();
    let invalid = u64::MAX - 31;
    let extent = super::extent(
        if suffix || fallback {
            start - 512
        } else {
            invalid
        },
        16,
    );
    engine.space = super::flat(&[(extent.start().get(), extent.len().get())]);
    let calls = Cell::new(0);
    let suffix_requests = if suffix {
        vec![block(u64::MAX)]
    } else {
        Vec::new()
    };
    let Err(failure) = engine.plan_blocks_with_fallback(
        block(if fallback { 32 } else { 16 }),
        &suffix_requests,
        || {
            calls.set(calls.get() + 1);
            Err(FormatError::InvalidChunkGeometry(
                "injected fallback failure",
            ))
        },
    ) else {
        panic!("expected a reservation failure")
    };
    let ReservationFailure { plan, error } = *failure;
    assert_eq!(calls.get(), u8::from(fallback));
    if fallback {
        assert!(
            matches!(
                error,
                Error::Format(FormatError::InvalidChunkGeometry(
                    "injected fallback failure"
                ))
            ),
            "{error:?}"
        );
    } else {
        assert!(
            matches!(error, Error::Format(FormatError::OffsetOverflow {offset, length}) if (offset,length) == if suffix {(end - 512,u64::MAX)} else {(invalid,512)}),
            "{error:?}"
        );
    }
    assert_eq!(plan.reservations().len(), usize::from(suffix));
    assert_eq!(
        plan.unresolved().map(|reservation| reservation.extent()),
        if suffix || fallback {
            None
        } else {
            Some(extent)
        }
    );
    if suffix {
        assert_eq!(plan.reservations()[0].range().start().get(), start);
    }
    drop(plan);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        if fallback { vec![extent] } else { Vec::new() }
    );
    assert_eq!(engine.image().read_exact_at(start, 64).unwrap(), [13; 64]);
}

#[rstest]
#[case::fresh_boundary(true)]
#[case::uncertain(false)]
fn preferred_reuse_preserves_session_eligibility(#[case] fresh: bool) {
    let (_directory, mut engine) = super::engine(FileSpaceStrategy::Page, 0);
    let start = engine.append(&[13; 128]).unwrap();
    engine.space = super::flat(&[(start, 64)]);
    if fresh {
        engine.fresh_page_boundary = Some(StoredAddress::new(engine.image.len()));
    } else {
        engine.context.allocations.make_uncertain();
    }
    let calls = Cell::new(0);
    match engine.plan_blocks_with_fallback(block(64), &[], || {
        calls.set(calls.get() + 1);
        Ok(vec![block(64)])
    }) {
        Ok((choice, plan)) => {
            assert!(fresh);
            assert_eq!(choice, ReservationChoice::Fallback);
            assert_eq!(plan.reservations()[0].range().start().get(), start + 128);
            drop(plan);
        }
        Err(failure) => {
            assert!(!fresh);
            assert!(
                matches!(failure.error, Error::AllocationPublicationUncertain),
                "{:?}",
                failure.error
            );
        }
    }
    assert_eq!(calls.get(), u8::from(fresh));
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [super::extent(start, 64)]
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::reuse(true, false)]
#[case::empty(false, false)]
#[case::one_appended(false, true)]
fn preferred_scalar_paths_avoid_batch_storage(#[case] reuse: bool, #[case] append: bool) {
    let (_directory, mut engine) = super::engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    let start = engine.append(&[13; 64]).unwrap();
    if reuse {
        engine.space = super::flat(&[(start, 64)]);
    }
    let (control, control_cost) =
        allocation::measure(&format!("preferred_control_{reuse}_{append}"), || {
            vec![7_u8; 1024]
        });
    assert_eq!(control, [7; 1024]);
    assert!(control_cost.bytes >= 1024);
    let ((choice, plan), cost) =
        allocation::measure(&format!("preferred_scalar_{reuse}_{append}"), || {
            engine
                .plan_blocks_with_fallback(block(64), &[], || {
                    Ok(if append { vec![block(64)] } else { Vec::new() })
                })
                .unwrap_or_else(|failure| panic!("{:?}", failure.error))
        });
    assert_eq!(
        choice,
        if reuse {
            ReservationChoice::Preferred
        } else {
            ReservationChoice::Fallback
        }
    );
    assert_eq!(plan.work(), ReservationWork::default());
    assert_eq!(plan.reservations().len(), usize::from(reuse || append));
    assert_eq!(cost.blocks, u64::from(append));
    assert_eq!(
        cost.bytes,
        if append {
            u64::try_from(core::mem::size_of::<BlockRequest>()).unwrap()
        } else {
            0
        }
    );
}

#[rstest]
#[case::success(false)]
#[case::partial_metadata_write(true)]
fn preferred_prefix_and_suffix_retain_complete_ownership(#[case] fail: bool) {
    let (_directory, mut engine) = super::engine(FileSpaceStrategy::FsmAggr, 0);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.aggregation = None;
    let start = engine.append(&[13; 512]).unwrap();
    engine.space = super::flat(&[(start, 64), (start + 128, 64), (start + 256, 64)]);
    let snapshot = engine.snapshot_free();
    let metadata = BlockRequest::complete(
        super::request(16, PageType::Meta),
        EncodingDependency::Allocation,
    );
    let (choice, mut plan) = engine
        .plan_blocks_with_fallback(block(64), &[metadata, metadata], || {
            panic!("a preferred hit must not prepare fallback requests")
        })
        .unwrap_or_else(|failure| panic!("{:?}", failure.error));
    assert_eq!(choice, ReservationChoice::Preferred);
    assert_eq!(
        plan.reservations()
            .iter()
            .map(|reservation| (
                reservation.range().start().get(),
                reservation.request().page_type()
            ))
            .collect::<Vec<_>>(),
        [
            (start, PageType::Raw),
            (start + 128, PageType::Meta),
            (start + 144, PageType::Meta)
        ]
    );
    assert_eq!(plan.work().indexed_extents, 2);
    assert_eq!(plan.work().restored_extents, 2);
    if fail {
        test_util::disk_log::fail_write_after(2, 3);
    }
    plan.write_next_with(|writer| {
        writer.write(&[17; 32])?;
        writer.write(&[19; 32])
    })
    .unwrap();
    let mut expected = [13; 512];
    expected[..32].fill(17);
    expected[32..64].fill(19);
    if fail {
        let error = plan.write_next(&[23; 16]).unwrap_err();
        let Error::Io(error) = error else {
            panic!("expected an injected I/O error, got {error:?}")
        };
        assert_eq!(error.kind(), ErrorKind::Other);
        assert_eq!(error.to_string(), "injected disk write failure");
        assert_eq!(plan.pending().len(), 2);
        drop(plan);
        expected[128..131].fill(23);
    } else {
        plan.write_next(&[23; 16]).unwrap();
        plan.write_next(&[29; 16]).unwrap();
        plan.finish().unwrap();
        expected[128..144].fill(23);
        expected[144..160].fill(29);
    }
    assert_eq!(engine.image.len(), start + 512);
    assert_eq!(engine.image().read_exact_at(start, 512).unwrap(), expected);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [
            super::extent(start + 160, 32),
            super::extent(start + 256, 64)
        ]
    );
    engine.restore_free(snapshot);
    assert_eq!(
        engine.space.reusable_sections().collect::<Vec<_>>(),
        [
            super::extent(start, 64),
            super::extent(start + 128, 64),
            super::extent(start + 256, 64)
        ]
    );
}

fn block(length: u64) -> BlockRequest {
    BlockRequest::complete(
        super::request(length, PageType::Raw),
        EncodingDependency::Independent,
    )
}
