use rstest::rstest;
use test_util_hdf5::page_object_headers::{self, Observation, SourceLayout};

use crate::PhysicalAddress;
use crate::edit::test_util;
use crate::source::Source;
use crate::{AttrValue, File};

#[rstest]
#[case::new_contiguous(Operation::Create, SourceLayout::Contiguous)]
#[case::new_chunked(Operation::Create, SourceLayout::Chunked)]
#[case::moving_contiguous(Operation::Overwrite, SourceLayout::Contiguous)]
#[case::moving_chunked(Operation::Overwrite, SourceLayout::Chunked)]
#[case::moving_compact(Operation::Compact, SourceLayout::Compact)]
#[case::attributes(Operation::Attribute, SourceLayout::Contiguous)]
#[case::append(Operation::Append, SourceLayout::Chunked)]
#[case::in_file_copy(Operation::Copy, SourceLayout::Contiguous)]
#[case::compact_copy(Operation::Copy, SourceLayout::Compact)]
#[case::chunked_copy(Operation::Copy, SourceLayout::Chunked)]
#[case::cross_file_chunked_copy(Operation::CopyFrom, SourceLayout::Chunked)]
#[case::cross_file_copy(Operation::CopyFrom, SourceLayout::Contiguous)]
fn ordinary_editor_headers_obey_page_geometry_and_native_reads(
    #[case] operation: Operation,
    #[case] layout: SourceLayout,
    #[values(512, 4_096)] page: u64,
    #[values(0, 8_192)] base: u64,
    #[values(48, 5_000)] padding: usize,
) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("edited.h5");
    let persist = !matches!(operation, Operation::Overwrite);
    page_object_headers::write_native(&path, page, base, layout, persist, padding);
    let file = File::open_rw(&path).unwrap();
    let parent = file.group("parent").unwrap();
    let nested = file.group("parent/nested").unwrap();
    let old_header = file
        .dataset("parent/nested/data")
        .unwrap()
        .header_address()
        .unwrap();
    let attr = AttrValue::U8Array(vec![31; padding]);
    file.root().set_attr("padding", attr.clone()).unwrap();
    parent.set_attr("padding", attr.clone()).unwrap();
    nested.set_attr("padding", attr.clone()).unwrap();
    let mut expected = vec![1i32, 2, 3, 4];
    let mut copied = false;
    match operation {
        Operation::Create => {
            nested
                .create_dataset("added", |builder| {
                    builder.with_i32_data(&[11, 12, 13, 14]);
                    builder.set_attr("padding", attr.clone());
                    if matches!(layout, SourceLayout::Chunked) {
                        builder.with_chunks(&[2]);
                    }
                })
                .unwrap();
        }
        Operation::Overwrite | Operation::Compact => {
            file.dataset("parent/nested/data")
                .unwrap()
                .write_staged(|builder| {
                    builder.with_i32_data(&[11, 12, 13, 14]);
                })
                .unwrap();
            expected = vec![11, 12, 13, 14];
        }
        Operation::Attribute => {
            file.dataset("parent/nested/data")
                .unwrap()
                .set_attr("padding", attr.clone())
                .unwrap();
        }
        Operation::Append => {
            file.dataset("parent/nested/data")
                .unwrap()
                .append_staged(|builder| {
                    builder.append_i32(&[5, 6]);
                })
                .unwrap();
            expected.extend([5, 6]);
        }
        Operation::Copy => {
            file.copy("parent", "copied").unwrap();
            copied = true;
        }
        Operation::CopyFrom => {
            let source_path = directory.path().join("source.h5");
            page_object_headers::write_native(&source_path, page, 0, layout, persist, padding);
            let source = File::open(&source_path).unwrap();
            file.copy_from(&source, "parent", "copied").unwrap();
            source.close().unwrap();
            copied = true;
        }
    }
    file.commit().unwrap();
    if matches!(
        operation,
        Operation::Overwrite | Operation::Compact | Operation::Attribute | Operation::Append
    ) {
        assert_ne!(
            file.dataset("parent/nested/data")
                .unwrap()
                .header_address()
                .unwrap(),
            old_header
        );
    }
    let mut headers = vec![
        file.root().header_address().unwrap(),
        parent.header_address().unwrap(),
        nested.header_address().unwrap(),
    ];
    if matches!(operation, Operation::Create) {
        headers.push(
            file.dataset("parent/nested/added")
                .unwrap()
                .header_address()
                .unwrap(),
        );
    } else if !copied {
        headers.push(
            file.dataset("parent/nested/data")
                .unwrap()
                .header_address()
                .unwrap(),
        );
    }
    if copied {
        headers.extend([
            file.group("copied").unwrap().header_address().unwrap(),
            file.group("copied/nested")
                .unwrap()
                .header_address()
                .unwrap(),
            file.dataset("copied/nested/data")
                .unwrap()
                .header_address()
                .unwrap(),
        ]);
    }
    drop(nested);
    drop(parent);
    file.close().unwrap();
    let engine = test_util::open_engine(&path);
    assert_eq!(engine.context.addresses().effective_base().get(), base);
    for (index, address) in headers.into_iter().enumerate() {
        let spans = engine
            .oh_chunk_spans(PhysicalAddress::new(address))
            .unwrap();
        assert_eq!(spans.len(), 1);
        let span = spans[0];
        if index < 3 {
            assert_eq!(
                span.length() >= page,
                padding >= usize::try_from(page).unwrap(),
                "{span:?}"
            );
        }
        let stored = span.start().get() - base;
        if span.length() < page {
            assert!(stored % page + span.length() <= page, "{span:?}");
        } else {
            assert_eq!(stored % page, 0, "{span:?}");
        }
        assert!(span.end().get() <= engine.image.len());
    }
    drop(engine);
    page_object_headers::assert_edited(
        &path,
        page,
        &expected,
        padding,
        match operation {
            Operation::Create => Observation::Added,
            Operation::Attribute => Observation::Attribute,
            Operation::Copy | Operation::CopyFrom => Observation::Copied,
            _ => Observation::Values,
        },
    );
    directory.close().unwrap();
}

#[derive(Clone, Copy)]
enum Operation {
    Append,
    Attribute,
    Compact,
    Copy,
    CopyFrom,
    Create,
    Overwrite,
}
