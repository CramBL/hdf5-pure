use crate::edit::test_util::durability;

#[test]
fn public_commit_reports_a_later_compaction_failure_without_replaying_edits() {
    durability::later_parent();
}
