use rstest::rstest;
use test_util_hdf5::__private::heap_members;

use crate::edit::analysis::AnalysisPhase;
use crate::edit::analysis::CollectionContent;
use crate::edit::analysis::CollectionProvenance;
use crate::edit::analysis::heap::associations::MemberPresence;
use crate::edit::analysis::heap::qualification::ParentUnavailable;
use crate::edit::analysis::physical::claims::domain::OwnerDomain;
use crate::edit::test_util as edit_test_util;
use crate::object_graph::ObjectId;

#[rstest]
#[case::ordinary(false)]
#[case::external_token(true)]
fn native_original_member_qualification_keeps_unmatched_members_independent(
    #[case] external_token: bool,
) {
    let path = test_util::temp::temp_path("native-collection-qualification.h5");
    let token = heap_members::write(&path, external_token);
    let original = std::fs::read(&path).unwrap();
    let engine = edit_test_util::open_engine(&path);
    let mut phase = AnalysisPhase::new(&engine);
    let root =
        ObjectId::resolve(engine.addresses(), engine.superblock().root_group_address).unwrap();
    let source = phase.reconstruction_source(&engine, root).unwrap();
    let candidates = phase.deleted_collections(&[root]);
    let facts = phase.qualify_collections(&source, candidates).unwrap();
    let Some(OwnerDomain::Available(proof)) = facts.domain() else {
        panic!("expected native original owner domain");
    };
    assert!(proof.matches(&engine, phase.physical_claims(&source).unwrap()));
    assert!(facts.predecessors_complete());
    assert_eq!(facts.parents().len(), 1);
    let (&address, parent) = facts.parents().first_key_value().unwrap();
    assert_eq!(parent.allocation(), Ok(()));
    assert_eq!(parent.members().len(), 2);
    assert!(
        parent
            .members()
            .values()
            .all(|presence| *presence == MemberPresence::Allocated)
    );
    let origin = CollectionProvenance {
        object: root,
        content: CollectionContent::Object,
    };
    let references = &facts.predecessors().unwrap()[&origin];
    assert_eq!(references.objects().len(), 2);
    for &id in references.objects() {
        assert_eq!(facts.member(&origin, id), Ok(()));
    }
    assert_eq!(
        facts.whole_parent(address),
        if external_token {
            Err(ParentUnavailable::Unmatched)
        } else {
            Ok(())
        }
    );
    let contents = facts.contents(&mut phase, address).unwrap();
    assert_eq!(contents.range(), parent.framing().unwrap());
    assert_eq!(contents.members().len(), if external_token { 3 } else { 2 });
    let associations = facts.associations(&mut phase).unwrap();
    assert_eq!(
        associations.collections()[&address].unmatched().len(),
        usize::from(external_token)
    );
    drop(phase);
    drop(engine);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    heap_members::assert_values(&path, token.as_ref());
}
