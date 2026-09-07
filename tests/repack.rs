//! Whole-file repack (issue #21): compaction, object dropping, fidelity of
//! survivors, and fail-loud refusal of features that cannot be reproduced.

use hdf5_pure::{
    AttrValue, Datatype, DatatypeByteOrder, FileBuilder, FileSpaceStrategy, LibVer, RepackOptions,
    ScaleOffset, repack,
};

use temp::temp_path;
use test_util::temp;

/// A variable-length attribute survives a repack still variable-length.
///
/// Repack reads every attribute and writes it back, so it can only preserve
/// what the reader reports. While a variable-length array read as a
/// fixed-width one, repack kept the strings and silently re-encoded the
/// layout — which is the encoding MATLAB and matio require for
/// `MATLAB_fields`. Only the VLEN read path yields `VarLenAsciiCharArray`, so
/// asserting the variant asserts the encoding, not just the values.
#[test]
fn carries_a_variable_length_attribute_without_re_encoding_it() {
    let src = temp_path("hdf5_pure_repack_vlen_attr_src.h5");
    let dst = temp_path("hdf5_pure_repack_vlen_attr_dst.h5");
    let fields: Vec<String> = vec!["x".into(), "y".into(), "velocity".into()];
    let mut b = FileBuilder::new();
    b.create_dataset("x").with_f64_data(&[1.0]);
    b.set_attr("fields", AttrValue::VarLenAsciiCharArray(fields.clone()));
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_f64_data(&[2.0]);
    g.set_attr("fields", AttrValue::VarLenAsciiCharArray(fields.clone()));
    b.add_group(g.finish());
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(
        f.root().attrs().unwrap().get("fields"),
        Some(&AttrValue::VarLenAsciiCharArray(fields.clone())),
        "root attribute must stay variable-length"
    );
    assert_eq!(
        f.group("grp").unwrap().attrs().unwrap().get("fields"),
        Some(&AttrValue::VarLenAsciiCharArray(fields)),
        "group attribute must stay variable-length"
    );
}

#[test]
fn drops_object_and_shrinks_file() {
    let src = temp_path("hdf5_pure_repack_drop_src.h5");
    let dst = temp_path("hdf5_pure_repack_drop_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("bulk").with_f64_data(&vec![9.0; 4096]);
    b.write(&src).unwrap();
    let src_size = std::fs::metadata(&src).unwrap().len();

    repack(&src, &dst, &RepackOptions::new().drop_path("bulk")).unwrap();

    let dst_size = std::fs::metadata(&dst).unwrap().len();
    assert!(
        dst_size < src_size,
        "dropping the bulk dataset should shrink the file (src {src_size}, dst {dst_size})"
    );

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(f.root().datasets().unwrap(), vec!["keep".to_string()]);
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(f.dataset("bulk").is_err());
}

#[test]
fn pure_compaction_preserves_everything() {
    let src = temp_path("hdf5_pure_repack_compact_src.h5");
    let dst = temp_path("hdf5_pure_repack_compact_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("ints").with_i64_data(&[10, 20, 30, 40]);
    b.create_dataset("floats").with_f64_data(&[1.5, 2.5, 3.5]);
    b.set_attr("title", AttrValue::String("experiment".to_string()));
    let mut g = b.create_group("grp");
    g.create_dataset("inner").with_f32_data(&[0.25, 0.5]);
    g.set_attr("units", AttrValue::AsciiString("m/s".to_string()));
    b.add_group(g.finish());
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(
        f.dataset("ints").unwrap().read_i64().unwrap(),
        vec![10, 20, 30, 40]
    );
    assert_eq!(
        f.dataset("floats").unwrap().read_f64().unwrap(),
        vec![1.5, 2.5, 3.5]
    );
    assert_eq!(
        f.dataset("grp/inner").unwrap().read_f32().unwrap(),
        vec![0.25, 0.5]
    );
    // Attributes carried over.
    let root_attrs = f.root().attrs().unwrap();
    assert_eq!(
        root_attrs.get("title"),
        Some(&AttrValue::String("experiment".to_string()))
    );
    let grp_attrs = f.group("grp").unwrap().attrs().unwrap();
    // The charset survives too: repack reads each attribute and writes it back,
    // so a fixed-width ASCII string used to come out UTF-8 once the reader had
    // dropped the distinction. It is written as ASCII here.
    assert_eq!(
        grp_attrs.get("units"),
        Some(&AttrValue::AsciiString("m/s".to_string()))
    );
}

#[test]
fn repacks_v1_symbol_table_source_with_attributes() {
    // `attrs.h5` is an older-format file (v0 superblock => v1 symbol-table
    // groups) carrying compact attributes. Repack now opens the source via the
    // streaming backend, so this drives v1 group traversal and compact attribute
    // reads end to end through the repack entry point (issues #82 / #27).
    let dst = temp_path("hdf5_pure_repack_v1_attrs_dst.h5");
    let src = "tests/data/unattributed/attrs.h5";

    let source = hdf5_pure::File::open(src).unwrap();
    let src_data = source.dataset("data").unwrap().read_f64().unwrap();
    let src_data_attrs = source.dataset("data").unwrap().attrs().unwrap();
    let src_root_attrs = source.root().attrs().unwrap();
    assert!(!src_data_attrs.is_empty() && !src_root_attrs.is_empty());

    repack(src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(f.dataset("data").unwrap().read_f64().unwrap(), src_data);
    assert_eq!(f.dataset("data").unwrap().attrs().unwrap(), src_data_attrs);
    assert_eq!(f.root().attrs().unwrap(), src_root_attrs);
}

#[test]
fn repacks_v1_nested_symbol_table_groups() {
    // `two_groups.h5` has v1 symbol-table groups nested under the root. Repacking
    // it exercises the streaming v1 B-tree/local-heap/SNOD traversal across
    // multiple groups and preserves the full subtree.
    let dst = temp_path("hdf5_pure_repack_v1_groups_dst.h5");
    let src = "tests/data/unattributed/two_groups.h5";

    repack(src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let mut groups = f.root().groups().unwrap();
    groups.sort();
    assert_eq!(groups, vec!["group1".to_string(), "group2".to_string()]);
    assert_eq!(
        f.dataset("group1/values").unwrap().read_i32().unwrap(),
        vec![10, 20, 30]
    );
    assert_eq!(
        f.group("group2").unwrap().datasets().unwrap(),
        vec!["temps".to_string()]
    );
}

#[test]
fn carries_dataset_attributes() {
    let src = temp_path("hdf5_pure_repack_dsattr_src.h5");
    let dst = temp_path("hdf5_pure_repack_dsattr_dst.h5");
    let mut b = FileBuilder::new();
    let ds = b.create_dataset("signal");
    ds.with_f64_data(&[1.0, 2.0, 3.0]);
    ds.set_attr("sample_rate", AttrValue::F64(44100.0));
    ds.set_attr("label", AttrValue::String("voltage".to_string()));
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let attrs = f.dataset("signal").unwrap().attrs().unwrap();
    assert_eq!(attrs.get("sample_rate"), Some(&AttrValue::F64(44100.0)));
    assert_eq!(
        attrs.get("label"),
        Some(&AttrValue::String("voltage".to_string()))
    );
    assert_eq!(
        f.dataset("signal").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn drops_whole_group_subtree() {
    let src = temp_path("hdf5_pure_repack_dropgrp_src.h5");
    let dst = temp_path("hdf5_pure_repack_dropgrp_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("root_ds").with_i32_data(&[1]);
    let mut g = b.create_group("doomed");
    g.create_dataset("a").with_i32_data(&[1, 2]);
    g.create_dataset("b").with_i32_data(&[3, 4]);
    b.add_group(g.finish());
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new().drop_path("/doomed")).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(f.root().datasets().unwrap(), vec!["root_ds".to_string()]);
    assert!(f.group("doomed").is_err());
    assert!(f.dataset("doomed/a").is_err());
}

#[test]
fn preserves_chunked_and_compressed_dataset() {
    let src = temp_path("hdf5_pure_repack_chunk_src.h5");
    let dst = temp_path("hdf5_pure_repack_chunk_dst.h5");
    let data: Vec<f64> = (0..2048).map(|i| i as f64 * 0.5).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[7]);
    b.create_dataset("comp")
        .with_f64_data(&data)
        .with_chunks(&[256])
        .with_shuffle()
        .with_deflate(6);
    b.write(&src).unwrap();

    // Drop nothing; just compact. The chunked+filtered dataset must survive with
    // byte-exact values.
    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(f.dataset("comp").unwrap().read_f64().unwrap(), data);
    assert_eq!(f.dataset("keep").unwrap().read_i32().unwrap(), vec![7]);
}

#[test]
fn preserves_multidim_and_maxshape() {
    let src = temp_path("hdf5_pure_repack_md_src.h5");
    let dst = temp_path("hdf5_pure_repack_md_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("grid")
        .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .with_shape(&[2, 3])
        .with_maxshape(&[u64::MAX, 3])
        .with_chunks(&[1, 3]);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let ds = f.dataset("grid").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![2, 3]);
    assert_eq!(ds.read_f64().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn roundtrips_integer_scale_offset() {
    // Integer scale-offset is lossless, so repack re-applies it (decompress +
    // recompress reconstructs the exact bytes). Highly compressible data (a tiny
    // value range) makes the filter's survival observable from the file size.
    let data: Vec<i32> = (0..4096).map(|i| i % 8).collect();

    let so_src = temp_path("hdf5_pure_repack_so_src.h5");
    let so_dst = temp_path("hdf5_pure_repack_so_dst.h5");
    let plain_src = temp_path("hdf5_pure_repack_soplain_src.h5");
    let plain_dst = temp_path("hdf5_pure_repack_soplain_dst.h5");

    let mut b = FileBuilder::new();
    b.create_dataset("vals")
        .with_i32_data(&data)
        .with_chunks(&[512])
        .with_scale_offset(ScaleOffset::Integer(0));
    b.write(&so_src).unwrap();

    // The same data, chunked but unfiltered, as a baseline for the size check.
    let mut p = FileBuilder::new();
    p.create_dataset("vals")
        .with_i32_data(&data)
        .with_chunks(&[512]);
    p.write(&plain_src).unwrap();

    repack(&so_src, &so_dst, &RepackOptions::new()).unwrap();
    repack(&plain_src, &plain_dst, &RepackOptions::new()).unwrap();

    // Values survive byte-exact.
    let f = hdf5_pure::File::open(&so_dst).unwrap();
    assert_eq!(f.dataset("vals").unwrap().read_i32().unwrap(), data);

    // The filter survived the repack: had it been dropped, the scale-offset copy
    // would be no smaller than the unfiltered one.
    let so_size = std::fs::metadata(&so_dst).unwrap().len();
    let plain_size = std::fs::metadata(&plain_dst).unwrap().len();
    assert!(
        so_size < plain_size,
        "repacked scale-offset file ({so_size}) should be smaller than the unfiltered repack ({plain_size}), proving the filter was re-applied"
    );
}

#[test]
fn roundtrips_lossy_float_scale_offset_verbatim() {
    // Float D-scale scale-offset is lossy, but a CHUNKED dataset's compressed
    // chunks are copied verbatim (never decoded), so repack reproduces the data
    // byte-exact instead of refusing. The values read back from the repacked file
    // must equal the values read back from the source (the lossy rounding is
    // baked into the stored bytes and carried through unchanged).
    let src = temp_path("hdf5_pure_repack_fso_src.h5");
    let dst = temp_path("hdf5_pure_repack_fso_dst.h5");
    let data: Vec<f64> = (0..1024).map(|i| (i as f64) * 0.01).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("vals")
        .with_f64_data(&data)
        .with_chunks(&[256])
        .with_scale_offset(ScaleOffset::FloatDScale(3));
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let src_vals = hdf5_pure::File::open(&src)
        .unwrap()
        .dataset("vals")
        .unwrap()
        .read_f64()
        .unwrap();
    let dst_f = hdf5_pure::File::open(&dst).unwrap();
    let dst_ds = dst_f.dataset("vals").unwrap();
    assert_eq!(
        dst_ds.read_f64().unwrap(),
        src_vals,
        "verbatim chunk copy must reproduce the (lossy-rounded) values exactly"
    );
    // The dataset is still chunked + filtered: a read loads the chunk index into
    // the per-dataset chunk cache (a contiguous dataset never would), and the
    // filter shrinks the file well below the raw 1024*8 bytes of element data.
    assert!(
        dst_ds.chunk_cache_stats().index_loaded(),
        "repacked dataset should still be chunked"
    );
    assert!(
        std::fs::metadata(&dst).unwrap().len() < 1024 * 8,
        "scale-offset filter should keep the repacked file below the raw data size"
    );
}

#[test]
fn roundtrips_opaque_and_bitfield_datatypes() {
    // Opaque and bit-field datatypes now serialize losslessly, so repack carries
    // them through byte-for-byte instead of refusing.
    let src = temp_path("hdf5_pure_repack_dt_src.h5");
    let dst = temp_path("hdf5_pure_repack_dt_dst.h5");

    let opaque_dt = Datatype::Opaque {
        size: 4,
        tag: b"rgba".to_vec(),
    };
    let opaque_raw: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 1, 2, 3, 4, 9, 8, 7, 6];
    let bitfield_dt = Datatype::BitField {
        size: 2,
        byte_order: DatatypeByteOrder::LittleEndian,
        bit_offset: 0,
        bit_precision: 12,
    };
    let bitfield_raw: Vec<u8> = vec![0x34, 0x12, 0xFF, 0x0F];

    let mut b = FileBuilder::new();
    b.create_dataset("blob")
        .with_raw_data(opaque_dt.clone(), opaque_raw.clone(), 3);
    b.create_dataset("flags")
        .with_raw_data(bitfield_dt.clone(), bitfield_raw.clone(), 2);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let blob = f.dataset("blob").unwrap();
    assert_eq!(blob.datatype().unwrap(), opaque_dt);
    assert_eq!(blob.read_raw().unwrap(), opaque_raw);
    let flags = f.dataset("flags").unwrap();
    assert_eq!(flags.datatype().unwrap(), bitfield_dt);
    assert_eq!(flags.read_raw().unwrap(), bitfield_raw);
}

#[test]
fn preserves_file_space_strategy() {
    let src = temp_path("hdf5_pure_repack_fss_src.h5");
    let dst = temp_path("hdf5_pure_repack_fss_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("keep").with_i32_data(&[1, 2, 3]);
    b.create_dataset("drop_me").with_f64_data(&vec![0.0; 1000]);
    b.with_file_space_strategy(FileSpaceStrategy::Page, false, 4)
        .with_file_space_page_size(8192);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new().drop_path("drop_me")).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    // The strategy (and its page size / threshold) carries forward; persist is
    // reset to false since the compact output has no free space to persist.
    assert_eq!(f.file_space_strategy(), Some(FileSpaceStrategy::Page));
    let info = f.file_space_info().unwrap();
    assert_eq!(info.page_size, 8192);
    assert_eq!(info.threshold, 4);
    assert!(!info.persist);
    assert_eq!(
        f.dataset("keep").unwrap().read_i32().unwrap(),
        vec![1, 2, 3]
    );
    assert!(f.dataset("drop_me").is_err());
}

#[test]
fn rejects_nonexistent_drop_path() {
    let src = temp_path("hdf5_pure_repack_baddrop_src.h5");
    let dst = temp_path("hdf5_pure_repack_baddrop_dst.h5");
    let mut b = FileBuilder::new();
    b.create_dataset("present").with_i32_data(&[1]);
    b.write(&src).unwrap();

    let err = repack(&src, &dst, &RepackOptions::new().drop_path("absent")).unwrap_err();
    assert!(
        matches!(err, hdf5_pure::Error::RepackUnsupported(_)),
        "expected RepackUnsupported, got {err:?}"
    );
    // Nothing should have been written to dst.
    assert!(
        !dst.exists(),
        "dst must not be created when the repack fails"
    );
}

#[test]
fn verbatim_chunk_copy_preserves_compressed_bytes() {
    // A chunked + deflate dataset: repack copies its compressed chunks verbatim,
    // so the values read back are byte-identical and the dataset stays chunked +
    // compressed (the file remains far smaller than the raw element bytes).
    let src = temp_path("hdf5_pure_repack_verbatim_src.h5");
    let dst = temp_path("hdf5_pure_repack_verbatim_dst.h5");
    // Highly compressible data so the filter's survival is observable by size.
    let data: Vec<i32> = (0..4096).map(|i| i % 4).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("vals")
        .with_i32_data(&data)
        .with_chunks(&[512])
        .with_shuffle()
        .with_deflate(6);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let ds = f.dataset("vals").unwrap();
    assert_eq!(
        ds.read_i32().unwrap(),
        data,
        "values must round-trip exactly"
    );
    assert!(
        ds.chunk_cache_stats().index_loaded(),
        "repacked dataset must still be chunked"
    );
    assert!(
        std::fs::metadata(&dst).unwrap().len() < 4096 * 4,
        "deflate filter must survive (file smaller than raw element bytes)"
    );
}

#[test]
fn repacks_multichunk_2d_fixed_array() {
    // A 2D dataset chunked into a 2x2 grid uses a v4 Fixed Array index. Repack's
    // verbatim path must lay the four chunks back in dense grid order so the
    // values round-trip exactly.
    let src = temp_path("hdf5_pure_repack_fa_src.h5");
    let dst = temp_path("hdf5_pure_repack_fa_dst.h5");
    // 4x4 grid, chunk 2x2 -> a 2x2 chunk grid (four chunks).
    let data: Vec<f64> = (0..16).map(|i| i as f64 * 1.5).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("grid")
        .with_f64_data(&data)
        .with_shape(&[4, 4])
        .with_chunks(&[2, 2])
        .with_deflate(4);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let ds = f.dataset("grid").unwrap();
    assert_eq!(ds.shape().unwrap(), vec![4, 4]);
    assert_eq!(ds.read_f64().unwrap(), data);
    assert!(ds.chunk_cache_stats().index_loaded());
}

#[test]
fn repacks_resizable_extensible_array() {
    // An unlimited-maxshape dataset uses a v4 Extensible Array index. Repack must
    // carry the maxshape through and reproduce the values exactly.
    let src = temp_path("hdf5_pure_repack_ea_src.h5");
    let dst = temp_path("hdf5_pure_repack_ea_dst.h5");
    let data: Vec<i64> = (0..1000).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("series")
        .with_i64_data(&data)
        .with_shape(&[1000])
        .with_maxshape(&[u64::MAX])
        .with_chunks(&[128])
        .with_deflate(3);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let ds = f.dataset("series").unwrap();
    assert_eq!(ds.read_i64().unwrap(), data);
    // Maxshape (resizability) is carried through.
    assert_eq!(ds.shape().unwrap(), vec![1000]);
    assert!(ds.chunk_cache_stats().index_loaded());
}

#[cfg(feature = "zfp")]
#[test]
fn roundtrips_zfp_verbatim() {
    // ZFP is a lossy filter this crate's read-raw path would refuse to re-encode,
    // but a chunked dataset's compressed chunks are copied verbatim, so repack
    // reproduces the stored (lossy-compressed) values byte-exact. The values read
    // back from the repacked file must equal those read back from the source.
    let src = temp_path("hdf5_pure_repack_zfp_src.h5");
    let dst = temp_path("hdf5_pure_repack_zfp_dst.h5");
    let data: Vec<f64> = (0..1024).map(|i| (i as f64).sin()).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("vals")
        .with_f64_data(&data)
        .with_chunks(&[256])
        .with_zfp(32.0);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let src_vals = hdf5_pure::File::open(&src)
        .unwrap()
        .dataset("vals")
        .unwrap()
        .read_f64()
        .unwrap();
    let dst_f = hdf5_pure::File::open(&dst).unwrap();
    let dst_ds = dst_f.dataset("vals").unwrap();
    assert_eq!(
        dst_ds.read_f64().unwrap(),
        src_vals,
        "verbatim ZFP chunk copy must reproduce the stored values exactly"
    );
    assert!(
        dst_ds.chunk_cache_stats().index_loaded(),
        "repacked ZFP dataset must still be chunked"
    );
}

#[test]
fn repacks_single_chunk_filtered_verbatim() {
    // A dataset whose single chunk covers the whole dataset uses the v4
    // single-chunk index. The verbatim path must carry the chunk's real filter
    // mask into that index and reproduce the values exactly.
    let src = temp_path("hdf5_pure_repack_single_src.h5");
    let dst = temp_path("hdf5_pure_repack_single_dst.h5");
    let data: Vec<i32> = (0..256).map(|i| i % 5).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("vals")
        .with_i32_data(&data)
        .with_chunks(&[256]) // one chunk covers all 256 elements
        .with_deflate(6);
    b.write(&src).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    let ds = f.dataset("vals").unwrap();
    assert_eq!(ds.read_i32().unwrap(), data);
    assert!(ds.chunk_cache_stats().index_loaded());
}

#[test]
fn repacks_chunked_dataset_from_a_userblock_file() {
    // The source carries a userblock (non-zero base address), so its chunk index
    // and chunk data are stored base-relative. Repack reads each chunk verbatim
    // from the source; it must apply the base address, or it reads the wrong bytes
    // and produces a corrupt copy.
    let src = temp_path("hdf5_pure_repack_ub_src.h5");
    let dst = temp_path("hdf5_pure_repack_ub_dst.h5");
    let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.25).collect();
    let mut b = FileBuilder::new();
    b.with_userblock(512);
    b.create_dataset("chk")
        .with_f64_data(&data)
        .with_shape(&[1000])
        .with_deflate(6);
    b.create_dataset("plain").with_f64_data(&[1.0, 2.0, 3.0]);
    std::fs::write(&src, b.finish().unwrap()).unwrap();

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    let f = hdf5_pure::File::open(&dst).unwrap();
    assert_eq!(f.dataset("chk").unwrap().read_f64().unwrap(), data);
    assert_eq!(
        f.dataset("plain").unwrap().read_f64().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

/// A repack carries the source's on-disk format forward.
///
/// The format is the one property whose loss is invisible: the repacked file
/// opens fine here, in `h5py`, and in every current-libhdf5 reader, and refuses
/// to open only in the old library the bound was set for. So it is asserted at
/// the byte the bound decides — the superblock version — rather than through a
/// read that would pass either way.
#[test]
fn preserves_the_source_on_disk_format() {
    let src = temp_path("hdf5_pure_repack_libver_src.h5");
    let dst = temp_path("hdf5_pure_repack_libver_dst.h5");
    let mut b = FileBuilder::new();
    b.with_libver_bounds(LibVer::Earliest, LibVer::V18);
    b.create_dataset("values").with_f64_data(&[1.0, 2.0, 3.0]);
    b.write(&src).unwrap();
    assert_eq!(superblock_version(&src), 2, "the source is the 1.8 format");

    repack(&src, &dst, &RepackOptions::new()).unwrap();

    assert_eq!(superblock_version(&dst), 2, "repack rewrote the format");
    assert_eq!(
        hdf5_pure::File::open(&dst)
            .unwrap()
            .dataset("values")
            .unwrap()
            .read_f64()
            .unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

/// The carried-forward format yields where the content leaves no choice.
///
/// HDF5 1.8 and earlier indexed chunks with a version 1 B-tree, which this crate
/// reads but does not write, so a pre-1.10 source holding a chunked dataset has
/// no older encoding to be repacked back into. Upgrading it is forced; refusing
/// would take repack away from most files the C library ever wrote, which are
/// chunked under a version 0 or 2 superblock.
///
/// `chunked_deflate.h5` is such a file — superblock 0, one chunked and deflated
/// dataset — so this is the forced case rather than a simulation of it. A source
/// this crate wrote would not be: its own chunked files are already 1.10, so
/// preserving and upgrading give the same answer and the test could not tell a
/// strict rule from the right one.
#[test]
fn upgrades_only_where_the_source_format_cannot_hold_the_content() {
    let dst = temp_path("hdf5_pure_repack_libver_forced_dst.h5");

    // Contiguous content under a version 0 superblock: floored at the oldest
    // format this crate writes, and no further.
    repack(
        "tests/data/unattributed/simple_dataset.h5",
        &dst,
        &RepackOptions::new(),
    )
    .unwrap();
    assert_eq!(
        superblock_version(&dst),
        2,
        "a contiguous repack upgraded past the oldest writable format"
    );

    // Chunked content under a version 0 superblock: upgraded rather than
    // refused, because there is no chunk index the older format can carry.
    let expected = hdf5_pure::File::open("tests/data/unattributed/chunked_deflate.h5")
        .unwrap()
        .dataset("data")
        .unwrap()
        .read_f64()
        .unwrap();
    repack(
        "tests/data/unattributed/chunked_deflate.h5",
        &dst,
        &RepackOptions::new(),
    )
    .unwrap();
    assert_eq!(superblock_version(&dst), 3);
    assert_eq!(
        hdf5_pure::File::open(&dst)
            .unwrap()
            .dataset("data")
            .unwrap()
            .read_f64()
            .unwrap(),
        expected
    );
}

/// An explicit bound is a guarantee, not a preference: content it cannot express
/// is refused rather than upgraded past it, which is the difference between it
/// and the carried-forward default above.
#[test]
fn an_explicit_bound_refuses_content_it_cannot_express() {
    let src = temp_path("hdf5_pure_repack_libver_chunked_src.h5");
    let dst = temp_path("hdf5_pure_repack_libver_chunked_dst.h5");
    let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("chk")
        .with_f64_data(&data)
        .with_shape(&[1000])
        .with_deflate(6);
    b.write(&src).unwrap();

    // Re-target down to 1.8: the chunked dataset cannot be written there.
    let to_v18 = RepackOptions::new().with_libver_bounds(LibVer::Earliest, LibVer::V18);
    assert!(
        matches!(
            repack(&src, &dst, &to_v18),
            Err(hdf5_pure::Error::Format(
                hdf5_pure::FormatError::LibverTooOldForContent { .. }
            ))
        ),
        "a chunked dataset must not be silently upgraded past the requested bound"
    );

    // And the same bound raised to 1.10 succeeds, so the refusal is about the
    // content and not about the option being set at all.
    let to_v110 = RepackOptions::new().with_libver_bounds(LibVer::Earliest, LibVer::V110);
    repack(&src, &dst, &to_v110).unwrap();
    assert_eq!(superblock_version(&dst), 3);
}

/// The superblock version byte of the file at `path`, found by scanning for the
/// signature so a userblock does not throw the offset off.
fn superblock_version(path: &std::path::Path) -> u8 {
    let bytes = std::fs::read(path).unwrap();
    let sig = bytes
        .windows(8)
        .position(|w| w == b"\x89HDF\r\n\x1a\n")
        .expect("the file carries an HDF5 signature");
    bytes[sig + 8]
}
