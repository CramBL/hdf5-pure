use std::fs;
use std::path::Path;

use hdf5_pure_space::__private::FreeClass;
use hdf5_pure_space::__private::PageType;
use rstest::rstest;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::free_space_churn;
use test_util_hdf5::free_space_churn::Base;
use test_util_hdf5::free_space_churn::Lifecycle;
use test_util_hdf5::free_space_churn::Strategy;
use test_util_hdf5::free_space_endurance;

use crate::FileLocking;
use crate::FileSpaceStrategy;
use crate::edit::SpaceOwnership;
use crate::edit::WriteEngine;

use super::page_managers;

#[derive(Clone, Copy, Debug)]
enum Workload {
    Continuous(Base),
    Endurance,
    Reopen(Base),
}

#[rstest]
#[case::continuous_zero(Workload::Continuous(Base::Zero))]
#[case::continuous_userblock(Workload::Continuous(Base::Userblock))]
#[case::reopened_zero(Workload::Reopen(Base::Zero))]
#[case::reopened_userblock(Workload::Reopen(Base::Userblock))]
#[case::endurance(Workload::Endurance)]
fn persisted_page_workloads_keep_live_free_and_padding_accounting_disjoint(
    #[case] workload: Workload,
) {
    let mut checkpoints = 0;
    let mut observe = |path: &Path, step: &str| {
        checkpoints += 1;
        let bytes = fs::read(path).unwrap();
        let engine = WriteEngine::open_with_locking(path, FileLocking::Enabled).unwrap();
        let SpaceOwnership {
            physical_size: _,
            logical_size,
            userblock_space,
            live_raw_space,
            live_metadata_space,
            free_space_manager_space,
            reusable_free_space,
            unclassified_space,
            analysis_complete,
            trailing_slack_space,
        } = engine.space_ownership().unwrap();
        assert!(analysis_complete, "{workload:?}/{step}");
        assert_eq!(trailing_slack_space, []);
        let length = |ranges: &[(u64, u64)]| ranges.iter().map(|&(_, len)| len).sum::<u64>();
        let raw = length(&live_raw_space);
        let metadata = length(&live_metadata_space);
        let managers = length(&free_space_manager_space);
        let free = length(&reusable_free_space);
        let unclassified = length(&unclassified_space);
        let base = length(&userblock_space);
        let last_raw = live_raw_space
            .iter()
            .map(|&(start, len)| start + len)
            .max()
            .unwrap_or(0);
        let last_metadata = live_metadata_space
            .iter()
            .map(|&(start, len)| start + len)
            .max()
            .unwrap_or(0);
        let last_manager = free_space_manager_space
            .iter()
            .map(|&(start, len)| start + len)
            .max()
            .unwrap_or(0);
        assert_eq!(
            base + raw + metadata + managers + free + unclassified,
            logical_size
        );
        let extension = engine
            .context
            .superblock()
            .superblock_extension_address
            .unwrap();
        let extension = engine.context.addresses().resolve(extension).unwrap();
        let extension_bytes: u64 = engine
            .oh_chunk_spans(extension.get())
            .unwrap()
            .iter()
            .map(|range| range.length())
            .sum();
        let (metadata_list, raw_list, unclassified_list) =
            engine.space.admitted_snapshot().sections().fold(
                (0_u64, 0_u64, 0_u64),
                |(metadata, raw, dead), (extent, class)| match class {
                    FreeClass::Dead => (metadata, raw, dead + extent.len()),
                    FreeClass::Page(PageType::Meta) => (metadata + extent.len(), raw, dead),
                    FreeClass::Page(PageType::Raw) => (metadata, raw + extent.len(), dead),
                },
            );
        assert_eq!(metadata_list + raw_list, free);
        eprintln!(
            "CLASS ACCOUNTING {workload:?}/{step}: extension={extension_bytes} other_metadata={} metadata_list={metadata_list} raw_list={raw_list} unclassified_list={unclassified_list}",
            metadata - extension_bytes
        );
        let (published_free, _) = page_managers::inspect(&engine, 4_096);
        assert_eq!(published_free, free);
        drop(engine);
        assert_eq!(fs::read(path).unwrap(), bytes);
        assert_eq!(
            file_space_oracle::assert_manager_metadata_bytes(path),
            managers
        );
        file_space_oracle::assert_persisted_sections(path);
        eprintln!(
            "ACCOUNTING {workload:?}/{step}: size={} base={base} raw={raw} metadata={metadata} managers={managers} free={free} unclassified={unclassified} last_raw={last_raw} last_metadata={last_metadata} last_manager={last_manager}",
            logical_size
        );
    };
    match workload {
        Workload::Continuous(base) => {
            free_space_churn::run_churn_observed(
                Lifecycle::ContinuousOpen,
                Strategy::Page,
                base,
                &mut observe,
            );
        }
        Workload::Endurance => {
            free_space_endurance::run_endurance_observed(FileSpaceStrategy::Page, &mut observe);
        }
        Workload::Reopen(base) => {
            free_space_churn::run_churn_observed(
                Lifecycle::ReopenBeforeFinalCheck,
                Strategy::Page,
                base,
                &mut observe,
            );
        }
    }
    assert_eq!(
        checkpoints,
        match workload {
            Workload::Continuous(_) => 1,
            Workload::Endurance => 17,
            Workload::Reopen(_) => 2,
        }
    );
}
