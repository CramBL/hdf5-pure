use hdf5_pure_format::Dataspace;
use hdf5_pure_format::DataspaceType;
use hdf5_pure_format::Datatype;
use hdf5_pure_format::DatatypeByteOrder;
use hdf5_pure_format::FillValueError;
use hdf5_pure_format::FilterDescription;
use hdf5_pure_format::FilterPipeline;
use hdf5_pure_format::FilterPipelineError;
use hdf5_pure_format::FixedPointLayout;
use hdf5_pure_format::MaxExtent;
use hdf5_pure_format::V3_FLAGS_DEFAULT;

#[test]
fn on_disk_messages_round_trip_through_the_public_api() {
    let datatype = Datatype::FixedPoint {
        size: 4,
        byte_order: DatatypeByteOrder::LittleEndian,
        layout: FixedPointLayout {
            signed: true,
            bit_offset: 0,
            bit_precision: 32,
        },
    };
    let datatype_bytes = hdf5_pure_format::serialize_datatype(&datatype);
    assert_eq!(
        hdf5_pure_format::element_size_usize(&datatype)
            .unwrap()
            .get(),
        4
    );
    assert_eq!(
        hdf5_pure_format::parse_datatype(&datatype_bytes),
        Ok((datatype, datatype_bytes.len()))
    );

    let dataspace = Dataspace {
        space_type: DataspaceType::Simple,
        rank: 1,
        dimensions: vec![3],
        max_dimensions: Some(vec![MaxExtent::Unlimited]),
    };
    let dataspace_bytes = dataspace.serialize(8);
    assert_eq!(Dataspace::parse(&dataspace_bytes, 8), Ok(dataspace));

    let pipeline = FilterPipeline {
        version: 2,
        filters: vec![FilterDescription {
            filter_id: 1,
            name: None,
            flags: 1,
            client_data: vec![6],
        }],
    };
    let pipeline_bytes = pipeline.serialize().unwrap();
    let parsed_pipeline: Result<FilterPipeline, FilterPipelineError> =
        FilterPipeline::parse(&pipeline_bytes);
    assert_eq!(parsed_pipeline, Ok(pipeline));

    let fill_message: Result<Vec<u8>, FillValueError> =
        hdf5_pure_format::fill_value_message_v3(None);
    assert_eq!(fill_message, Ok(vec![3, V3_FLAGS_DEFAULT]));
}
