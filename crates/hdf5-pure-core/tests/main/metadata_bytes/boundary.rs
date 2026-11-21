use hdf5_pure_core::MetadataBytes;

trait WorkspaceOperations {
    fn borrowed();
    fn owned();
    fn range(&self);
    fn region(&self);
    fn subregion(&self);
}

impl WorkspaceOperations for MetadataBytes<'_> {
    fn borrowed() {}
    fn owned() {}
    fn range(&self) {}
    fn region(&self) {}
    fn subregion(&self) {}
}

#[test]
fn workspace_operations_resolve_through_an_external_trait() {
    let bytes = MetadataBytes::from_slice(512, &[1, 2, 3]).unwrap();
    let (): () = MetadataBytes::borrowed();
    let (): () = MetadataBytes::owned();
    let (): () = bytes.range();
    let (): () = bytes.region();
    let (): () = bytes.subregion();
    assert_eq!(bytes.offset(), 512);
    assert_eq!(bytes.as_ref(), &[1, 2, 3]);
}
