#![cfg(feature = "__hdf5-1.10")]

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use hdf5_pure::AttrValue;
use hdf5_pure::File;
use hdf5_pure::FileAccessProperties;
use hdf5_pure::FileSpacePageSize;
use hdf5_pure::FileSpaceStrategy;
use hdf5_pure::MemoryStrategy;
use hdf5_pure::MessageType;
use hdf5_pure::PersistedFreeSpacePolicy;
use hdf5_pure::SyncPolicy;
use hdf5_pure_format::__private::AccessMode;
use hdf5_pure_format::__private::AttributeInfoMessage;
use hdf5_pure_format::__private::FormatWidths;
use hdf5_pure_format::__private::ObjectHeader;
use rstest::rstest;
use test_util::snapshot::text::Report;
use test_util::snapshot::text::Value;
use test_util_hdf5::file_space::persisted::PersistedStrategy;
use test_util_hdf5::file_space_oracle;
use test_util_hdf5::global_heap::attributes as fixture;
use test_util_hdf5::global_heap::attributes::Label;
use test_util_hdf5::global_heap::attributes::Owner;
use test_util_hdf5::global_heap::attributes::Storage;
use test_util_hdf5::storage;

use crate::test_util::file_space;

#[rstest]
fn attribute_collection_cycles_reuse_storage_through_persistent_reopen(
    #[values(
        Session::BoundedFsm,
        Session::BoundedPage,
        Session::MirroredFsm,
        Session::MirroredPage
    )]
    session: Session,
    #[values(Owner::Dataset, Owner::Group, Owner::Root)] owner: Owner,
    #[values(Storage::Compact, Storage::Dense)] storage: Storage,
) {
    let path = test_util::temp::temp_path("attribute-collection-cycles.h5");
    fixture::write_persisted(&path, owner, storage, session.strategy());
    fixture::assert_contents(&path, owner, storage, Label::Ascii("before"));
    assert_storage(&path, owner, storage);
    let mut file = File::open_rw_with_options(&path, session.access()).unwrap();
    let settings = file.file_space_settings().unwrap().unwrap();
    assert_eq!(settings.threshold(), 1);
    assert!(settings.persist());
    assert_eq!(
        settings.strategy(),
        match session.strategy() {
            PersistedStrategy::FsmAggr => FileSpaceStrategy::FsmAggr,
            PersistedStrategy::Page => FileSpaceStrategy::Page,
        }
    );
    let mut checkpoints = vec![Checkpoint::read("native fixture", &file)];
    let mut sizes = Vec::new();
    let mut complements = Vec::new();
    for round in 0..REPLACEMENTS {
        let value = payload(round);
        set(&file, owner, AttrValue::VarLenAsciiString(value.clone()));
        file.commit().unwrap();
        assert_eq!(
            attributes(&file, owner)["label"],
            AttrValue::VarLenAsciiString(value.clone())
        );
        let state = Checkpoint::read("replacement", &file);
        assert!(state.file_bytes - state.reusable_bytes >= u64::try_from(PAYLOAD_BYTES).unwrap());
        sizes.push(state.file_bytes);
        complements.push(state.file_bytes - state.reusable_bytes);
        if round == 3 {
            checkpoints.push(Checkpoint::read("four replacements", &file));
        }
        if round == 5 {
            file.close().unwrap();
            file_space_oracle::assert_persisted_sections(&path);
            fixture::assert_contents(&path, owner, storage, Label::Ascii(&value));
            file = File::open_rw_with_options(&path, session.access()).unwrap();
            checkpoints.push(Checkpoint::read("six and reopen", &file));
        }
    }
    let early_peak = sizes[..4].iter().copied().max().unwrap();
    let late_peak = sizes[4..].iter().copied().max().unwrap();
    assert!(
        late_peak <= early_peak + 2 * u64::try_from(PAYLOAD_BYTES).unwrap() + 4 * PAGE_BYTES,
        "file sizes: {sizes:?}"
    );
    let late_complements = &complements[4..];
    assert!(
        late_complements.iter().max().unwrap() - late_complements.iter().min().unwrap()
            <= 4 * PAGE_BYTES,
        "nonfree-byte observations: {complements:?}"
    );
    checkpoints.push(Checkpoint::read("twelve replacements", &file));
    remove(&file, owner);
    file.commit().unwrap();
    assert!(!attributes(&file, owner).contains_key("label"));
    checkpoints.push(Checkpoint::read("removed", &file));
    file.close().unwrap();
    file_space_oracle::assert_persisted_sections(&path);
    fixture::assert_contents(&path, owner, storage, Label::Missing);
    let file = File::open_rw_with_options(&path, session.access()).unwrap();
    set(
        &file,
        owner,
        AttrValue::VarLenAsciiString(payload(REPLACEMENTS)),
    );
    file.commit().unwrap();
    checkpoints.push(Checkpoint::read("restored after reopen", &file));
    let final_size = file.file_size();
    assert!(final_size <= late_peak + u64::try_from(PAYLOAD_BYTES).unwrap() + 4 * PAGE_BYTES);
    file.close().unwrap();
    fixture::assert_contents(&path, owner, storage, Label::Ascii(&payload(REPLACEMENTS)));
    let observation = file_space_oracle::observe(&path, "attribute collection reuse", "restored");
    if let Some(name) = session.snapshot(owner, storage) {
        let mut report = Report::new();
        report.table(
            "Configuration",
            ["Property", "Value"],
            [
                [
                    Value::text("Memory strategy"),
                    Value::text(format!("{:?}", session.memory())),
                ],
                [
                    Value::text("Attribute owner"),
                    Value::text(format!("{owner:?}")),
                ],
                [
                    Value::text("Initial attribute storage"),
                    Value::text(format!("{storage:?}")),
                ],
            ],
        );
        report.metrics(
            "Workload",
            "Value",
            [
                ("Replacement commits", u64::try_from(REPLACEMENTS).unwrap()),
                (
                    "Payload bytes per replacement",
                    u64::try_from(PAYLOAD_BYTES).unwrap(),
                ),
                ("Guard bytes", u64::try_from(fixture::GUARD_BYTES).unwrap()),
            ],
        );
        report.table(
            "Confirmed session state",
            ["Phase", "File bytes", "Reusable bytes", "Reusable extents"],
            checkpoints.iter().map(|row| {
                [
                    Value::text(row.phase),
                    Value::Number(row.file_bytes),
                    Value::Number(row.reusable_bytes),
                    Value::Number(row.reusable_extents),
                ]
            }),
        );
        file_space::snapshot(
            name,
            &format!("{}\n\n{}", report.finish(), observation.render()),
        );
    }
}

struct Checkpoint {
    phase: &'static str,
    file_bytes: u64,
    reusable_bytes: u64,
    reusable_extents: u64,
}

impl Checkpoint {
    fn read(phase: &'static str, file: &File) -> Self {
        let stats = file.reusable_space_stats().unwrap();
        let reusable = file.reusable_free_sections().unwrap();
        test_util_hdf5::assert_free_sections_consistent!(
            &stats,
            &reusable,
            hdf5_pure::PhysicalAddress
        );
        Self {
            phase,
            file_bytes: stats.logical_size(),
            reusable_bytes: stats.reusable_free_bytes(),
            reusable_extents: u64::try_from(reusable.section_count()).unwrap(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Session {
    BoundedFsm,
    BoundedPage,
    MirroredFsm,
    MirroredPage,
}

impl Session {
    fn strategy(self) -> PersistedStrategy {
        match self {
            Self::BoundedFsm | Self::MirroredFsm => PersistedStrategy::FsmAggr,
            Self::BoundedPage | Self::MirroredPage => PersistedStrategy::Page,
        }
    }

    fn memory(self) -> MemoryStrategy {
        match self {
            Self::BoundedFsm | Self::BoundedPage => MemoryStrategy::Bounded,
            Self::MirroredFsm | Self::MirroredPage => MemoryStrategy::Mirrored,
        }
    }

    fn access(self) -> FileAccessProperties {
        FileAccessProperties::new()
            .with_memory_strategy(self.memory())
            .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict)
            .with_sync_policy(SyncPolicy::OnClose)
    }

    fn snapshot(self, owner: Owner, storage: Storage) -> Option<&'static str> {
        match (self, owner, storage) {
            (Self::BoundedFsm, Owner::Root, Storage::Compact) => {
                Some("attribute_reuse_fsm_bounded")
            }
            (Self::MirroredFsm, Owner::Dataset, Storage::Dense) => {
                Some("attribute_reuse_fsm_mirrored")
            }
            (Self::BoundedPage, Owner::Group, Storage::Dense) => {
                Some("attribute_reuse_page_bounded")
            }
            (Self::MirroredPage, Owner::Root, Storage::Compact) => {
                Some("attribute_reuse_page_mirrored")
            }
            _ => None,
        }
    }
}

fn assert_storage(path: &Path, owner: Owner, storage: Storage) {
    let bytes = fs::read(path).unwrap();
    let file = File::open(path).unwrap();
    let superblock = file.superblock();
    let address = match owner {
        Owner::Root => file.resolve_address(superblock.root_group_address).unwrap(),
        Owner::Dataset | Owner::Group => storage::root_object_address(&file, &bytes, owner.path()),
    };
    let header = ObjectHeader::parse_with_base_and_widths(
        &bytes,
        AccessMode::ReadOnly,
        usize::try_from(address.get()).unwrap(),
        FormatWidths {
            offsets: superblock.offset_size,
            lengths: superblock.length_size,
        },
        superblock.base_address,
    )
    .unwrap();
    let info = header
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::ATTRIBUTE_INFO)
        .map(|message| {
            AttributeInfoMessage::parse_with_offset_width(&message.data, superblock.offset_size)
                .unwrap()
        });
    let dense = matches!(storage, Storage::Dense);
    assert_eq!(
        info.as_ref()
            .is_some_and(|info| info.fractal_heap_address.is_some()),
        dense
    );
    assert_eq!(
        info.as_ref()
            .is_some_and(|info| info.btree_name_index_address.is_some()),
        dense
    );
    assert_eq!(
        header
            .messages
            .iter()
            .filter(|message| message.msg_type == MessageType::ATTRIBUTE)
            .count(),
        usize::from(!dense)
    );
    file.close().unwrap();
}

fn set(file: &File, owner: Owner, value: AttrValue) {
    match owner {
        Owner::Dataset => file
            .dataset(owner.path())
            .unwrap()
            .set_attr("label", value)
            .unwrap(),
        Owner::Group => file
            .group(owner.path())
            .unwrap()
            .set_attr("label", value)
            .unwrap(),
        Owner::Root => file.root().set_attr("label", value).unwrap(),
    }
}

fn remove(file: &File, owner: Owner) {
    match owner {
        Owner::Dataset => file
            .dataset(owner.path())
            .unwrap()
            .remove_attr("label")
            .unwrap(),
        Owner::Group => file
            .group(owner.path())
            .unwrap()
            .remove_attr("label")
            .unwrap(),
        Owner::Root => file.root().remove_attr("label").unwrap(),
    }
}

fn attributes(file: &File, owner: Owner) -> HashMap<String, AttrValue> {
    match owner {
        Owner::Dataset => file.dataset(owner.path()).unwrap().attrs().unwrap(),
        Owner::Group => file.group(owner.path()).unwrap().attrs().unwrap(),
        Owner::Root => file.root().attrs().unwrap(),
    }
}

fn payload(round: usize) -> String {
    let prefix = format!("round {round}: ");
    format!("{prefix}{}", "x".repeat(PAYLOAD_BYTES - prefix.len()))
}

const REPLACEMENTS: usize = 12;
const PAYLOAD_BYTES: usize = 24_000;
const PAGE_BYTES: u64 = FileSpacePageSize::DEFAULT.get();
