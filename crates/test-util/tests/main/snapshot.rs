use test_util::snapshot::text::{Report, Value};

#[test]
fn report_aligns_columns_and_groups_decimal_values() {
    let mut report = Report::new();
    report.table(
        "Stored allocations",
        ["Class", "Address", "Bytes"],
        [
            [
                Value::text("Raw"),
                Value::Number(1_234_567),
                Value::Number(0),
            ],
            [
                Value::text("Metadata"),
                Value::Number(8),
                Value::Number(65_536),
            ],
        ],
    );
    report.table("Returned allocations", ["Address", "Bytes"], []);
    report.metrics("Summary", "Bytes", [("Total", 1_234_567)]);
    assert_eq!(
        report.finish(),
        concat!(
            "Stored allocations\n",
            "  Class       Address   Bytes\n",
            "  --------  ---------  ------\n",
            "  Raw       1,234,567       0\n",
            "  Metadata          8  65,536\n",
            "\n",
            "Returned allocations\n",
            "  Address  Bytes\n",
            "  -------  -----\n",
            "  (none)\n",
            "\n",
            "Summary\n",
            "  Metric      Bytes\n",
            "  ------  ---------\n",
            "  Total   1,234,567",
        )
    );
}

#[test]
fn report_escapes_layout_controls_in_text_cells() {
    let mut report = Report::new();
    report.table(
        "Text",
        ["Value"],
        [[Value::text("first\nsecond\rthird\tfourth")]],
    );
    assert_eq!(
        report.finish(),
        "Text\n  Value\n  ----------------------------\n  first\\nsecond\\rthird\\tfourth"
    );
}
