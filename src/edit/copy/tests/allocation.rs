use hdf5_pure_format::__private::DenseAttrPlan;
use rstest::rstest;
use test_util::allocation;

use crate::access_mode::AccessMode;
use crate::address::{AddressContext, StoredAddressExt};
use crate::attribute::AttributeMessage;
use crate::chunked_write::{ChunkMeta, ChunkProvider};
use crate::data_layout::ChunkedLayoutFlags;
use crate::edit::dataset::ChunkGeometry;
use crate::error::{Error, FormatError};
use crate::group::v2 as group_v2;
use crate::message_type::MessageType;
use crate::object_header::ObjectHeader;
use crate::object_path::ObjectPathBuf;
use crate::reader::OpenedSource;
use crate::source::BytesSource;
use crate::superblock;
use crate::width::{FormatWidths, LengthWidth, OffsetWidth};
use crate::{AttrValue, FileBuilder, StoredAddress};

use super::super::{CopyNode, CopyOutput, WriteEngine};

#[rstest]
#[case::narrow_offsets(OffsetWidth::Two, LengthWidth::Eight, 256, ["copy_header_narrow_offsets", "copy_header_clone_narrow_offsets"])]
#[case::narrow_lengths(OffsetWidth::Eight, LengthWidth::Two, 256, ["copy_header_narrow_lengths", "copy_header_clone_narrow_lengths"])]
#[case::small(OffsetWidth::Eight, LengthWidth::Eight, 32, ["copy_header_small", "copy_header_clone_small"])]
#[case::large(OffsetWidth::Eight, LengthWidth::Eight, 8192, ["copy_header_large", "copy_header_clone_large"])]
fn chunked_copy_bounds_header_allocations(
    #[case] offsets: OffsetWidth,
    #[case] lengths: LengthWidth,
    #[case] padding: usize,
    #[case] names: [&str; 2],
) {
    let widths = FormatWidths { offsets, lengths };
    let mut builder = FileBuilder::new();
    builder
        .with_offset_width(offsets)
        .with_length_width(lengths);
    builder
        .create_dataset("values")
        .with_u32_data(&[1, 2])
        .with_chunks(&[1])
        .set_attr("padding", AttrValue::String("x".repeat(padding)));
    let bytes = builder.finish().unwrap();
    let source = BytesSource::new(bytes.as_slice());
    let located = superblock::read(&source).unwrap();
    let addresses = AddressContext::from_effective_base(located.superblock.base_address);
    let opened = OpenedSource::for_test(&source, addresses);
    let address = group_v2::resolve_path_any_from_source(
        &opened,
        AccessMode::ReadOnly,
        &located.superblock,
        widths,
        &ObjectPathBuf::parse("values").as_path(),
    )
    .unwrap();
    let tree =
        WriteEngine::read_copy_subtree(&opened, AccessMode::ReadOnly, address, false, widths)
            .unwrap();
    let CopyNode::DatasetChunked {
        geometry,
        flags,
        pipeline_message,
        meta,
        ..
    } = &tree.root
    else {
        panic!("the fixture must be chunked");
    };
    assert!(!tree.has_dense_attributes());
    assert!(tree.root.region().len() >= padding);
    let layout = geometry
        .plan_verbatim(
            widths,
            *flags,
            pipeline_message.as_deref(),
            meta,
            StoredAddress::new(16_384),
        )
        .unwrap()
        .layout_message;
    let header_len = tree
        .root
        .region()
        .props()
        .encode_header(tree.root.region())
        .len();
    let mut output = HeaderOutput {
        widths,
        layout: Some(layout.clone()),
        header: Vec::with_capacity(header_len),
    };
    let (address, measured) =
        allocation::measure(names[0], || tree.root.emit(&mut output, &[]).unwrap());
    assert_eq!(address, StoredAddress::new(32_768));
    assert!(output.layout.is_none());
    assert_eq!(output.header.len(), header_len);
    let parsed =
        ObjectHeader::parse_with_widths(&output.header, AccessMode::ReadOnly, 0, widths).unwrap();
    assert_eq!(
        parsed
            .messages
            .iter()
            .find(|message| message.msg_type == MessageType::DATA_LAYOUT)
            .unwrap()
            .data,
        layout
    );
    let attribute = parsed
        .messages
        .iter()
        .find(|message| message.msg_type == MessageType::ATTRIBUTE)
        .unwrap();
    let attribute = AttributeMessage::parse_with_widths(&attribute.data, widths).unwrap();
    assert_eq!(attribute.name, "padding");
    assert_eq!(attribute.raw_data, vec![b'x'; padding]);
    let mut control = HeaderOutput {
        widths,
        layout: Some(layout),
        header: Vec::with_capacity(header_len),
    };
    let (address, copied) = allocation::measure(names[1], || {
        let retained = tree.root.region().clone();
        let address = tree.root.emit(&mut control, &[]).unwrap();
        std::hint::black_box(retained);
        address
    });
    assert_eq!(address, StoredAddress::new(32_768));
    assert_eq!(control.header, output.header);
    assert_eq!(copied.blocks, 4, "{copied}");
    assert_eq!(
        copied.bytes,
        measured.bytes + u64::try_from(tree.root.region().len()).unwrap(),
        "{measured}; control {copied}"
    );
    assert_eq!(measured.blocks, 3, "{measured}");
    assert_eq!(
        measured.bytes,
        2 * u64::try_from(tree.root.region().len()).unwrap() + u64::try_from(header_len).unwrap(),
        "{measured}"
    );
    assert_eq!(measured.live_bytes, 0, "{measured}");
}

/// A header capture sink with reserved capacity and a previously encoded chunk layout.
struct HeaderOutput {
    widths: FormatWidths,
    layout: Option<Vec<u8>>,
    header: Vec<u8>,
}

impl CopyOutput for HeaderOutput {
    fn output_widths(&self) -> FormatWidths {
        self.widths
    }

    fn place_data(&mut self, _data: &[u8]) -> Result<StoredAddress, Error> {
        Err(
            FormatError::Internal("a chunk-header measurement places no contiguous data".into())
                .into(),
        )
    }

    fn place_chunks(
        &mut self,
        _geometry: &ChunkGeometry,
        _flags: ChunkedLayoutFlags,
        _pipeline: Option<&[u8]>,
        _meta: &[ChunkMeta],
        _chunks: &dyn ChunkProvider,
    ) -> Result<Vec<u8>, Error> {
        self.layout.take().ok_or_else(|| {
            FormatError::Internal("a chunk-header measurement emits one layout".into()).into()
        })
    }

    fn place_attributes(&mut self, _plan: DenseAttrPlan) -> Result<Vec<u8>, Error> {
        Err(
            FormatError::Internal("a chunk-header measurement has compact attributes".into())
                .into(),
        )
    }

    fn place_header(&mut self, header: &[u8]) -> Result<StoredAddress, Error> {
        self.header.extend_from_slice(header);
        Ok(StoredAddress::new(32_768))
    }
}
