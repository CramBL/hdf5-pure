use hdf5_pure_core::__private::{FileRange, MetadataBytesExt};
use hdf5_pure_core::{MetadataBytes, PhysicalAddress};

#[test]
fn workspace_subregions_borrow_from_the_owner() {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
    let pointer = data.as_ptr();
    let capacity = data.capacity();
    let range = FileRange::new(PhysicalAddress::new(u64::MAX - 6), 6).unwrap();
    let owner = MetadataBytes::owned(range, data).unwrap();
    assert_eq!(owner.as_ptr(), pointer);
    let middle = owner
        .region()
        .subregion(FileRange::new(PhysicalAddress::new(u64::MAX - 5), 4).unwrap())
        .unwrap();
    let nested = middle
        .subregion(FileRange::new(PhysicalAddress::new(u64::MAX - 4), 2).unwrap())
        .unwrap();
    let selected = MetadataBytes::from_region(nested);
    assert_eq!(selected.offset(), u64::MAX - 4);
    assert_eq!(selected.as_ref(), &[3, 4]);
    assert_eq!(selected.as_ptr(), owner.get(2..).unwrap().as_ptr());
    assert!(selected.shared().is_none());
    assert_eq!(selected.retained_bytes(), 0);
    assert_eq!(owner.retained_bytes(), capacity);
}

#[test]
fn stable_ownership_accessors_preserve_the_payload_allocation() {
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&[1, 2, 3]);
    let pointer = data.as_ptr();
    let capacity = data.capacity();
    let owner = MetadataBytes::from_vec(512, data).unwrap();
    let retained = owner.shared().unwrap();
    let cloned = owner.clone();
    drop(owner);
    drop(cloned);
    assert_eq!(retained.offset(), 512);
    assert_eq!(retained.retained_bytes(), capacity);
    let data = retained.into_vec();
    assert_eq!(data, vec![1, 2, 3]);
    assert_eq!(data.as_ptr(), pointer);
    assert_eq!(data.capacity(), capacity);
}

#[test]
fn borrowed_ownership_accessors_keep_the_borrow() {
    let data = [1, 2, 3];
    let owner = MetadataBytes::from_slice(512, &data).unwrap();
    assert_eq!(owner.as_ptr(), data.as_ptr());
    assert!(owner.shared().is_none());
    assert_eq!(owner.retained_bytes(), 0);
    assert_eq!(owner.into_vec(), data);
}
