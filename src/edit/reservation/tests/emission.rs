use hdf5_pure_core::__private::NonZeroByteLength;
use hdf5_pure_space::__private::AllocationClass;
use rstest::rstest;
use test_util::disk_log;

use crate::chunk_index_inplace::Store;
use crate::image::WriteBuffering;

use super::*;

#[rstest]
#[case::append(FileSpaceStrategy::None)]
#[case::aggregates(FileSpaceStrategy::Aggr)]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn streamed_blocks_preserve_complete_capacity_and_neighbors(
    #[case] strategy: FileSpaceStrategy,
    #[values(false, true)] reused: bool,
    #[values(0, 4096)] base: u64,
) {
    let (_directory, mut engine) = engine(strategy, base);
    engine.aggregation = None;
    let prior = engine.image.len();
    if reused {
        engine.append(&[13; 128]).unwrap();
        engine.space = flat(&[(prior + 16 - base, 64)]);
    }
    let blocks = [BlockRequest::new(
        request(64, PageType::Raw),
        32,
        EncodingDependency::Allocation,
    )
    .unwrap()];
    let mut plan = engine
        .plan_blocks(&blocks)
        .unwrap_or_else(|failure| panic!("planning failed: {:?}", failure.error));
    let allocation = plan.reservations()[0].range();
    let start = allocation.start();
    assert_eq!(allocation.length(), 64);
    assert_eq!(
        emit(&mut plan, |address, write| {
            assert_eq!(address, start);
            write(&[17; 15])?;
            write(&[])?;
            write(&[19; 17])
        })
        .unwrap(),
        start
    );
    plan.finish().unwrap();
    assert_eq!(
        engine.image().read_exact_at(start.get(), 32).unwrap(),
        [vec![17; 15], vec![19; 17]].concat()
    );
    if reused {
        assert_eq!(start.get(), prior + 16);
        assert_eq!(engine.image.len(), prior + 128);
        assert_eq!(engine.image().read_exact_at(prior, 16).unwrap(), [13; 16]);
        assert_eq!(
            engine.image().read_exact_at(prior + 48, 80).unwrap(),
            [13; 80]
        );
    } else {
        assert_eq!(engine.image.len(), start.get() + 64);
        assert_eq!(
            engine.image().read_exact_at(start.get() + 32, 32).unwrap(),
            [0; 32]
        );
    }
}

#[rstest]
#[case::short(EmissionFailure::Short)]
#[case::long(EmissionFailure::Long)]
#[case::producer(EmissionFailure::Producer)]
#[case::suppressed_sink_error(EmissionFailure::SuppressedSinkError)]
#[case::io(EmissionFailure::Io)]
fn failed_streams_keep_reserved_storage_unavailable_and_stop_later_writes(
    #[case] failure: EmissionFailure,
) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    engine.aggregation = None;
    let start = engine.append(&[13; 128]).unwrap();
    engine.space = flat(&[(start + 16, 16), (start + 64, 16)]);
    let requests = [request(16, PageType::Raw), request(16, PageType::Raw)];
    let mut plan = planned(&mut engine, &requests);
    let error = emit(&mut plan, |address, write| {
        assert_eq!(address.get(), start + 16);
        write(&[17; 8])?;
        match failure {
            EmissionFailure::Short => Ok(()),
            EmissionFailure::Long => write(&[19; 9]),
            EmissionFailure::Producer => Err(FormatError::LengthTooLarge { length: 65_536, length_size: 2 }.into()),
            EmissionFailure::SuppressedSinkError => {
                let error = write(&[19; 9]).unwrap_err();
                assert!(matches!(&error, Error::Format(FormatError::SerializationError(reason)) if reason == "a placement reserved 16 bytes but built 17"), "{error:?}");
                let error = write(&[23; 8]).unwrap_err();
                assert!(matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reserved encoding has an earlier failed write"), "{error:?}");
                Ok(())
            }
            EmissionFailure::Io => {
                disk_log::fail_write_after(0, 3);
                write(&[19; 8])
            }
        }
    }).unwrap_err();
    match failure {
        EmissionFailure::Short => assert!(
            matches!(&error, Error::Format(FormatError::SerializationError(reason)) if reason == "a placement reserved 16 bytes but built 8"),
            "{error:?}"
        ),
        EmissionFailure::Long => assert!(
            matches!(&error, Error::Format(FormatError::SerializationError(reason)) if reason == "a placement reserved 16 bytes but built 17"),
            "{error:?}"
        ),
        EmissionFailure::Producer => assert!(
            matches!(
                error,
                Error::Format(FormatError::LengthTooLarge {
                    length: 65_536,
                    length_size: 2
                })
            ),
            "{error:?}"
        ),
        EmissionFailure::SuppressedSinkError => assert!(
            matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reserved encoding has an earlier failed write"),
            "{error:?}"
        ),
        EmissionFailure::Io => {
            let Error::Io(error) = error else {
                panic!("expected injected I/O error, got {error:?}")
            };
            assert_eq!(error.kind(), ErrorKind::Other);
            assert_eq!(error.to_string(), "injected disk write failure");
        }
    }
    assert_eq!(plan.next, 0);
    assert!(plan.failed);
    let error = emit(&mut plan, |_, _| {
        panic!("failed plan invokes another producer")
    })
    .unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reservation plan has an earlier failed write"),
        "{error:?}"
    );
    let error = plan.finish().unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reservation plan has unwritten allocations"),
        "{error:?}"
    );
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), []);
    assert_eq!(engine.image.len(), start + 128);
    assert_eq!(engine.image().read_exact_at(start, 16).unwrap(), [13; 16]);
    assert_eq!(
        engine.image().read_exact_at(start + 16, 8).unwrap(),
        [17; 8]
    );
    assert_eq!(
        engine.image().read_exact_at(start + 32, 96).unwrap(),
        [13; 96]
    );
}

#[rstest]
#[case::append(FileSpaceStrategy::None)]
#[case::aggregate(FileSpaceStrategy::Aggr)]
#[case::page(FileSpaceStrategy::Page)]
fn streamed_padding_failures_prevent_plan_completion(#[case] strategy: FileSpaceStrategy) {
    let (_directory, mut engine) = engine(strategy, 0);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let blocks = [BlockRequest::new(
        request(64, PageType::Raw),
        16,
        EncodingDependency::Allocation,
    )
    .unwrap()];
    let mut plan = engine
        .plan_blocks(&blocks)
        .unwrap_or_else(|failure| panic!("planning failed: {:?}", failure.error));
    let start = plan.reservations()[0].range().start();
    let error = emit(&mut plan, |address, write| {
        assert_eq!(address, start);
        write(&[61; 16])?;
        disk_log::fail_write_after(0, 0);
        Ok(())
    })
    .unwrap_err();
    let Error::Io(error) = error else {
        panic!("expected padding I/O error, got {error:?}")
    };
    assert_eq!(error.kind(), ErrorKind::Other);
    assert_eq!(error.to_string(), "injected disk write failure");
    assert_eq!(plan.next, 0);
    assert!(plan.failed);
    assert_eq!(plan.engine.image.len(), start.get() + 16);
    assert_eq!(
        plan.engine.image().read_exact_at(start.get(), 16).unwrap(),
        [61; 16]
    );
    let error = plan.finish().unwrap_err();
    assert!(
        matches!(&error,Error::Format(FormatError::Internal(reason)) if reason == "reservation plan has unwritten allocations"),
        "{error:?}"
    );
}

#[rstest]
#[case::flat(FileSpaceStrategy::FsmAggr)]
#[case::page(FileSpaceStrategy::Page)]
fn persistent_stream_failures_restore_prior_root_allocator_and_eof(
    #[case] strategy: FileSpaceStrategy,
    #[values(false, true)] after_write: bool,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("restore-emission.h5");
    let mut builder = FileBuilder::new();
    builder.with_file_space_settings(FileSpaceSettings::new(strategy).with_persistence(true));
    builder.create_dataset("live").with_i32_data(&[2, 3, 5]);
    builder.write(&path).unwrap();
    let mut engine = edit_test_util::open_engine(&path);
    let before = engine.disk_bytes().unwrap();
    let free = engine.space.reusable_sections().collect::<Vec<_>>();
    let error = match engine.store().allocate_emitted(
        NonZeroByteLength::new(128).unwrap(),
        AllocationClass::Metadata,
        |_, write| {
            if after_write {
                write(&[29; 64])?;
            }
            Err(FormatError::LengthTooLarge {
                length: 65_536,
                length_size: 2,
            }
            .into())
        },
    ) {
        Ok(_) => panic!("producer failure returns an allocation"),
        Err(error) => error,
    };
    assert!(
        matches!(
            error,
            Error::Format(FormatError::LengthTooLarge {
                length: 65_536,
                length_size: 2
            })
        ),
        "{error:?}"
    );
    assert_eq!(engine.disk_bytes().unwrap(), before);
    assert_eq!(engine.space.reusable_sections().collect::<Vec<_>>(), free);
    let allocation = engine
        .store()
        .allocate_emitted(
            NonZeroByteLength::new(128).unwrap(),
            AllocationClass::Metadata,
            |_, write| {
                write(&[31; 64])?;
                write(&[37; 64])
            },
        )
        .unwrap();
    let range = allocation.range();
    let mut store = engine.store();
    let error = store.write_at(0, &[0]).unwrap_err();
    assert!(
        matches!(&error, Error::Format(FormatError::Internal(reason)) if reason == "reference publication precedes allocation ownership publication"),
        "{error:?}"
    );
    let confirmed = store
        .publish_allocations()
        .unwrap()
        .confirm(allocation)
        .unwrap();
    assert_eq!(
        store.allocation_publication().require(&confirmed).unwrap(),
        range
    );
    assert_eq!(
        engine
            .image()
            .read_exact_at(range.start().get(), 128)
            .unwrap(),
        [vec![31; 64], vec![37; 64]].concat()
    );
    drop(engine);
    let file = crate::File::open(&path).unwrap();
    assert_eq!(file.dataset("live").unwrap().read_i32().unwrap(), [2, 3, 5]);
    drop(file);
}

#[rstest]
#[case::append(FileSpaceStrategy::None)]
#[case::aggregates(FileSpaceStrategy::Aggr)]
fn nonpersistent_stream_failures_block_dependent_publication(#[case] strategy: FileSpaceStrategy) {
    let (_directory, mut engine) = engine(strategy, 0);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let prior = engine.disk_bytes().unwrap();
    let error = match engine.store().allocate_emitted(
        NonZeroByteLength::new(16).unwrap(),
        AllocationClass::Metadata,
        |_, write| {
            write(&[53; 8])?;
            Err(FormatError::LengthTooLarge {
                length: 65_536,
                length_size: 2,
            }
            .into())
        },
    ) {
        Ok(_) => panic!("producer failure returns an allocation"),
        Err(error) => error,
    };
    assert!(
        matches!(
            error,
            Error::Format(FormatError::LengthTooLarge {
                length: 65_536,
                length_size: 2
            })
        ),
        "{error:?}"
    );
    let mut store = engine.store();
    let error = match store.allocate_emitted(
        NonZeroByteLength::new(16).unwrap(),
        AllocationClass::Metadata,
        |_, _| panic!("uncertain allocation invokes a producer"),
    ) {
        Ok(_) => panic!("uncertain state returns another allocation"),
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    let error = match store.publish_allocations() {
        Ok(_) => panic!("uncertain allocation publishes ownership"),
        Err(error) => error,
    };
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    let error = store.write_at(0, &[0]).unwrap_err();
    assert!(
        matches!(error, Error::AllocationPublicationUncertain),
        "{error:?}"
    );
    assert_eq!(&engine.disk_bytes().unwrap()[..prior.len()], &prior);
}

#[rstest]
#[case::appended(false)]
#[case::reused(true)]
fn empty_encodings_preserve_complete_reserved_capacity(#[case] reused: bool) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::FsmAggr, 0);
    engine.aggregation = None;
    let start = engine.image.len();
    if reused {
        engine.append(&[59; 64]).unwrap();
        engine.space = flat(&[(start, 64)]);
    }
    let requests = [BlockRequest::new(
        request(64, PageType::Raw),
        0,
        EncodingDependency::Independent,
    )
    .unwrap()];
    let mut plan = engine
        .plan_blocks(&requests)
        .unwrap_or_else(|failure| panic!("planning failed: {:?}", failure.error));
    assert_eq!(
        emit(&mut plan, |address, write| {
            assert_eq!(address.get(), start);
            write(&[])
        })
        .unwrap()
        .get(),
        start
    );
    plan.finish().unwrap();
    assert_eq!(engine.image.len(), start + 64);
    assert_eq!(
        engine.image().read_exact_at(start, 64).unwrap(),
        [if reused { 59 } else { 0 }; 64]
    );
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[rstest]
#[case::small(1_024, "finite_stream_small", "finite_stream_small_control")]
#[case::large(1_024_000, "finite_stream_large", "finite_stream_large_control")]
fn streamed_fixed_arrays_use_bounded_heap(
    #[case] entries: u64,
    #[case] region: &'static str,
    #[case] control_region: &'static str,
) {
    use hdf5_pure_format::__private::FixedArrayHeaderLocator;
    use hdf5_pure_format::__private::FixedArrayPlan;
    use test_util::allocation;
    use test_util::fixed_array as fixture;
    use test_util::widths::Widths;

    let (_directory, mut engine) = engine(FileSpaceStrategy::None, 0);
    assert!(engine.image.as_slice().is_none());
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let widths = engine.context.widths();
    let plan = FixedArrayPlan::new(entries, 16, widths, false).unwrap();
    let length = plan.total_len();
    let header_length = plan.block_lengths().next().unwrap();
    let addresses = engine.context.addresses();
    let start = engine.image.len();
    let (control, positive) = allocation::measure(control_region, || {
        std::hint::black_box(vec![0_u8; usize::try_from(length).unwrap()])
    });
    assert!(positive.blocks > 0, "{positive}");
    assert!(positive.peak_bytes >= length, "{positive}");
    drop(control);
    let write_bound = usize::try_from(length.div_ceil(4_096) + 4).unwrap();
    assert!(engine.image.reserve_write_observations(write_bound));
    let prior_writes = engine.image.issued_writes();
    let (allocation, measured) = allocation::measure(region, || {
        engine
            .store()
            .allocate_emitted(
                NonZeroByteLength::new(length).unwrap(),
                AllocationClass::Metadata,
                |address, emit| {
                    assert_eq!(address.get(), start);
                    let header = addresses.relative(address)?;
                    let placed = plan.place(header, header.checked_add(header_length)?)?;
                    emit(&placed.serialize_header()?)?;
                    placed.emit_empty_data_block::<Error>(emit)
                },
            )
            .unwrap()
    });
    assert!(engine.image.issued_writes() - prior_writes <= u64::try_from(write_bound).unwrap());
    assert!(measured.peak_bytes < 4_096, "{measured}");
    assert!(measured.bytes < 4_096, "{measured}");
    assert!(measured.live_bytes < 4_096, "{measured}");
    assert_eq!(allocation.range().start().get(), start);
    assert_eq!(allocation.range().length(), length);
    let stored = addresses.relative(allocation.range().start()).unwrap();
    let expected_widths = Widths::EIGHT;
    let mut expected = fixture::header(
        expected_widths,
        fixture::Client::Plain,
        plan.page_bits(),
        entries,
        Some(stored.get() + header_length),
    );
    expected.extend_from_slice(&fixture::empty_data_block(
        expected_widths,
        fixture::Client::Plain,
        plan.page_bits(),
        usize::try_from(entries).unwrap(),
        stored.get(),
    ));
    assert_eq!(
        engine
            .image()
            .read_exact_at(start, usize::try_from(length).unwrap())
            .unwrap(),
        expected
    );
    let locator = FixedArrayHeaderLocator::new(widths, BaseAddress::new(0), stored);
    let parsed = locator
        .parse(
            engine
                .image()
                .read_metadata(start, usize::try_from(header_length).unwrap())
                .unwrap()
                .region(),
        )
        .unwrap();
    assert_eq!(parsed.entries(), entries);
    assert_eq!(parsed.page_bits(), plan.page_bits());
    assert_eq!(engine.image.len(), start + length);
}

#[rstest]
#[case::small(16)]
#[case::large(1_024)]
fn streamed_reservations_issue_each_fragment_once(#[case] fragments: u64) {
    let (_directory, mut engine) = engine(FileSpaceStrategy::None, 0);
    engine
        .set_write_buffering(WriteBuffering::Unbuffered)
        .unwrap();
    let start = engine.image.len();
    disk_log::start();
    let allocation = engine
        .store()
        .allocate_emitted(
            NonZeroByteLength::new(fragments * 1_024).unwrap(),
            AllocationClass::Metadata,
            |address, write| {
                assert_eq!(address.get(), start);
                for _ in 0..fragments {
                    write(&[41; 1_024])?;
                }
                Ok(())
            },
        )
        .unwrap();
    let operations = disk_log::take();
    assert_eq!(operations.len(), usize::try_from(fragments).unwrap());
    for (index, operation) in operations.into_iter().enumerate() {
        assert_eq!(
            operation,
            disk_log::DiskOp::Write {
                offset: start + u64::try_from(index).unwrap() * 1_024,
                bytes: vec![41; 1_024]
            }
        );
    }
    assert_eq!(allocation.range().length(), fragments * 1_024);
    assert_eq!(engine.image.len(), start + fragments * 1_024);
    assert_eq!(
        engine
            .image()
            .read_exact_at(start, usize::try_from(fragments * 1_024).unwrap())
            .unwrap(),
        vec![41; usize::try_from(fragments * 1_024).unwrap()]
    );
}

#[test]
fn streamed_aggregate_capacity_is_padded_once_after_all_fragments() {
    let (_directory, mut engine) = engine(FileSpaceStrategy::Aggr, 0);
    let start = engine.image.len();
    observations::reset_zeros();
    let allocation = engine
        .store()
        .allocate_emitted(
            NonZeroByteLength::new(16).unwrap(),
            AllocationClass::Raw,
            |address, write| {
                assert_eq!(address.get(), start);
                write(&[43; 7])?;
                assert_eq!(observations::zero_work(), ZeroWork::default());
                write(&[47; 9])?;
                assert_eq!(observations::zero_work(), ZeroWork::default());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(allocation.range().length(), 16);
    assert_eq!(engine.image.len(), start + 2_048);
    assert_eq!(
        observations::zero_work(),
        ZeroWork {
            calls: 1,
            bytes: 2_032,
            scratch_peak: 4_096
        }
    );
    assert_eq!(
        engine.image().read_exact_at(start, 16).unwrap(),
        [vec![43; 7], vec![47; 9]].concat()
    );
    assert_eq!(
        engine.image().read_exact_at(start + 16, 2_032).unwrap(),
        [0; 2_032]
    );
}

#[derive(Clone, Copy)]
enum EmissionFailure {
    Io,
    Long,
    Producer,
    Short,
    SuppressedSinkError,
}

fn emit(
    plan: &mut ReservationPlan<'_>,
    producer: impl FnOnce(
        PhysicalAddress,
        &mut dyn FnMut(&[u8]) -> Result<(), Error>,
    ) -> Result<(), Error>,
) -> Result<PhysicalAddress, Error> {
    let address = plan.next_reservation()?.block.encoding().start();
    plan.write_next_with(|writer| producer(address, &mut |bytes| writer.write(bytes)))
}
