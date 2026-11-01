use rstest::rstest;

use super::super::{CreationProperties, FreeSpaceOracleCheckpoint, PropertyValues, SectionMap};

#[rstest]
#[case::equal(vec![(4096, 1024)], vec![(4096, 1024)], 1024, "true", "yes", "0", ["1,024", "1,024", "1,024"])]
#[case::unequal(vec![(4096, 1024)], vec![(4096, 1000)], 999, "false", "NO", "-1", ["1,024", "1,000", "999"])]
#[case::missing_reference(vec![(4096, 1024)], Vec::new(), 0, "false", "NO", "0", ["1,024", "0", "0"])]
#[case::missing_rust(Vec::new(), vec![(4096, 1024)], 1023, "false", "NO", "-1", ["0", "1,024", "1,023"])]
fn paired_report_preserves_geometry_totals_and_missing_rows(
    #[case] rust: Vec<(u64, u64)>,
    #[case] c: Vec<(u64, u64)>,
    #[case] reported: u64,
    #[case] equal: &str,
    #[case] matched: &str,
    #[case] difference: &str,
    #[case] expected_totals: [&str; 3],
) {
    let c_bytes = c.iter().map(|section| section.1).sum::<u64>();
    let rust_empty = rust.is_empty();
    let c_empty = c.is_empty();
    let observation = checkpoint(rust, c, reported);
    let rendered = observation.render();
    assert_eq!(
        rows(&rendered, "checkpoint"),
        [vec!["case", "known-geometry"], vec!["checkpoint", "closed"]]
    );
    assert_eq!(
        rows(&rendered, "file-space properties"),
        [
            vec!["strategy", "FSM_AGGR", "FSM_AGGR"],
            vec!["persistence", "true", "true"],
            vec!["threshold", "(bytes)", "1,234,567", "1,234,567"],
            vec!["page", "size", "(bytes)", "not", "stored", "4,096"],
            vec!["base", "/", "userblock", "(bytes)", "0", "0"],
        ]
    );
    assert_eq!(
        rows(&rendered, "file sizes (absolute bytes)"),
        [
            vec!["hdf5-pure", "stored", "EOF", "8,192"],
            vec!["libhdf5", "H5Fget_filesize", "8,192"],
            vec!["physical", "file", "8,192"],
        ]
    );
    assert_eq!(
        rows(&rendered, "comparison"),
        [vec!["exact", "section", "map", "equal", equal]]
    );
    let row = rendered
        .lines()
        .find(|line| line.trim_start().starts_with("0 "))
        .unwrap();
    let cells = row.split_whitespace().collect::<Vec<_>>();
    assert_eq!(
        cells,
        [
            "0",
            if rust_empty {
                "<missing>"
            } else {
                "0x00001000"
            },
            if rust_empty {
                "<missing>"
            } else {
                "0x00000400"
            },
            if c_empty { "<missing>" } else { "0x00001000" },
            if c_empty {
                "<missing>"
            } else if c_bytes == 1000 {
                "0x000003e8"
            } else {
                "0x00000400"
            },
            matched,
        ]
    );
    assert_eq!(
        rows(&rendered, "free-space totals"),
        [
            vec![
                "section",
                "count",
                if rust_empty { "0" } else { "1" },
                if c_empty { "0" } else { "1" }
            ],
            vec!["section", "bytes", expected_totals[0], expected_totals[1]],
            vec!["H5Fget_freespace", "-", expected_totals[2]],
            vec!["reported", "-", "section", "bytes", "-", difference],
        ]
    );
}

#[test]
fn paired_report_identifies_empty_section_maps() {
    let rendered = checkpoint(Vec::new(), Vec::new(), 0).render();
    let sections = rendered
        .split("free-space sections (stored addresses, byte lengths)\n")
        .nth(1)
        .unwrap()
        .split("\n\n")
        .next()
        .unwrap();
    assert_eq!(sections.lines().last(), Some("  (none)"));
    assert!(
        rendered.contains("exact section map equal  true"),
        "{rendered}"
    );
}

fn checkpoint(
    rust: Vec<(u64, u64)>,
    c: Vec<(u64, u64)>,
    reported: u64,
) -> FreeSpaceOracleCheckpoint {
    FreeSpaceOracleCheckpoint {
        case: "known-geometry".into(),
        step: "closed".into(),
        rust: SectionMap::new("hdf5-pure", rust, 8192).unwrap(),
        c: SectionMap::new("libhdf5", c, 8192).unwrap(),
        c_reported_freespace: reported,
        rust_logical_size: 8192,
        c_file_size: 8192,
        physical_file_size: 8192,
        base: 0,
        properties: CreationProperties {
            rust: PropertyValues {
                strategy: "FSM_AGGR",
                persist: true,
                threshold: 1_234_567,
                page_size: None,
                base: 0,
            },
            c: PropertyValues {
                strategy: "FSM_AGGR",
                persist: true,
                threshold: 1_234_567,
                page_size: Some(4096),
                base: 0,
            },
        },
    }
}

fn rows<'a>(report: &'a str, title: &str) -> Vec<Vec<&'a str>> {
    report
        .lines()
        .skip_while(|line| *line != title)
        .skip(3)
        .take_while(|line| !line.is_empty())
        .map(|line| line.split_whitespace().collect())
        .collect()
}
