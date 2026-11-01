//! Huge-object ownership and read crosschecks against libhdf5.

use hdf5_pure_format::__private::{FormatWidths, FractalHeapHeader, FractalHeapIdLayout};
use hdf5_pure_space::__private::Extent;
use test_util_hdf5::heap_allocations::{AllocationKind, Heap};

use crate::PhysicalAddress;
use crate::access_mode::AccessMode;
use crate::address::{
    AddressContext, BaseAddress, BaseAddressExt, StoredAddress, StoredAddressExt,
};
use crate::convert::Narrow;
use crate::fractal_heap;
use crate::fractal_heap::HeapObjectReader;
use crate::fractal_heap::HeapSource;
use crate::fractal_heap_storage::{self, FractalHeapAllocation};
use crate::source::BytesSource;
use crate::source::PhysicalSource;
use crate::width::{LengthWidth, OffsetWidth};

#[cfg(feature = "deflate")]
#[rstest::rstest]
#[case::indirect(false)]
#[case::direct(true)]
fn c_written_filtered_huge_objects_have_exact_storage_and_contents(#[case] direct: bool) {
    let path = test_util::temp::temp_path("filtered_huge.h5");
    test_util_hdf5::fractal_heap::write_filtered_huge(&path, direct);
    let bytes = std::fs::read(&path).unwrap();
    let file = crate::reader::File::from_bytes(bytes.clone()).unwrap();
    let address = file.dataset("heap_address").unwrap().read_u64().unwrap()[0];
    let id = file.dataset("heap_id").unwrap().read_u8().unwrap();
    let header = FractalHeapHeader::parse(&bytes, address.to_usize().unwrap(), 8, 8).unwrap();
    let widths = FormatWidths {
        offsets: OffsetWidth::Eight,
        lengths: LengthWidth::Eight,
    };
    let inspected = Heap::new(&bytes, address, &header);
    let storage = fractal_heap_storage::collect_fractal_heap_storage_from_source(
        &BytesSource::new(&bytes),
        AddressContext::from_effective_base(BaseAddress::ZERO),
        StoredAddress::new(address),
        widths,
    )
    .unwrap();
    let expected: Vec<_> = inspected
        .allocations
        .iter()
        .map(|item| {
            let (at, length) = item.extent();
            let kind = match item.kind() {
                AllocationKind::Header => FractalHeapAllocation::Header,
                AllocationKind::HugeIndex => FractalHeapAllocation::HugeObjectIndex,
                AllocationKind::HugeObject => FractalHeapAllocation::HugeObject,
                kind => panic!("unexpected heap allocation: {kind:?}"),
            };
            (
                Extent::new(
                    file.addresses().relative(PhysicalAddress::new(at)).unwrap(),
                    length,
                )
                .unwrap(),
                kind,
            )
        })
        .collect();
    assert_eq!(storage.allocations(), expected);
    let expected: Vec<_> = (0..12_000)
        .map(|index| u8::try_from(index % 251).unwrap())
        .collect();
    file.with_opened_source(|opened| {
        let physical = PhysicalSource::new(opened);
        let stored = physical.stored();
        let (actual, range) = fractal_heap::read_header_and_range_from_source(
            &HeapSource::new(&stored),
            StoredAddress::new(address),
            widths,
        )
        .unwrap();
        assert_eq!(actual, header);
        assert_eq!(range.start().get(), address);
        let id = FractalHeapIdLayout::new(&header, OffsetWidth::Eight, LengthWidth::Eight)
            .parse(&id)
            .unwrap();
        assert_eq!(
            HeapObjectReader::new(
                HeapSource::new(&stored),
                &header,
                StoredAddress::new(address),
                AccessMode::ReadOnly,
                widths,
            )
            .read(id)
            .unwrap(),
            expected,
        );
        let resident = PhysicalSource::new(opened)
            .resident()
            .unwrap()
            .stored()
            .unwrap();
        assert_eq!(
            HeapObjectReader::new(
                HeapSource::resident(resident),
                &header,
                StoredAddress::new(address),
                AccessMode::ReadOnly,
                widths,
            )
            .read(id)
            .unwrap(),
            expected,
        );
    });
    file.close().unwrap();
}
