//! Formatting helpers for text snapshots.

/// A text report of titled tables.
///
/// Tables are separated by one blank line.
///
/// # Examples
///
/// ```
/// use test_util::snapshot::{Report, Value};
///
/// let mut report = Report::new();
/// report.table(
///     "Allocations",
///     ["Class", "Bytes"],
///     [[Value::text("Raw"), Value::Number(1_234)]],
/// );
/// report.metrics("Summary", "Count", [("Allocations", 1)]);
/// assert_eq!(
///     report.finish(),
///     concat!(
///         "Allocations\n",
///         "  Class  Bytes\n",
///         "  -----  -----\n",
///         "  Raw    1,234\n",
///         "\n",
///         "Summary\n",
///         "  Metric       Count\n",
///         "  -----------  -----\n",
///         "  Allocations      1",
///     ),
/// );
/// ```
#[derive(Default)]
pub struct Report {
    output: String,
}

impl Report {
    /// Creates an empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a table of metric labels and numeric values.
    ///
    /// The column headings are "Metric" and `value_label`.
    pub fn metrics<'a>(
        &mut self,
        title: &str,
        value_label: &str,
        rows: impl IntoIterator<Item = (&'a str, u64)>,
    ) {
        self.table(
            title,
            ["Metric", value_label],
            rows.into_iter()
                .map(|(label, value)| [Value::text(label), Value::Number(value)]),
        );
    }

    /// Appends a titled table with column widths fitted to its headings and rendered values.
    ///
    /// Columns containing at least one [`Value::Number`] are right-aligned. Other columns are
    /// left-aligned. Newlines, carriage returns, and tabs in titles, headings, and text cells are
    /// escaped. Empty tables display "(none)" below the column headings.
    ///
    /// # Panics
    ///
    /// Panics if `N` is zero.
    pub fn table<const N: usize>(
        &mut self,
        title: &str,
        headers: [&str; N],
        rows: impl IntoIterator<Item = [Value; N]>,
    ) {
        assert!(N > 0, "a snapshot table has at least one column");
        let rows: Vec<_> = rows.into_iter().collect();
        let numeric = core::array::from_fn(|column| {
            rows.iter()
                .any(|row| matches!(row[column], Value::Number(_)))
        });
        let headers = headers.map(single_line);
        let rows: Vec<_> = rows.into_iter().map(|row| row.map(Value::render)).collect();
        let widths = core::array::from_fn(|column| {
            rows.iter()
                .map(|row| row[column].chars().count())
                .chain([headers[column].chars().count()])
                .max()
                .unwrap_or(0)
        });
        if !self.output.is_empty() {
            self.output.push('\n');
        }
        self.output.push_str(&single_line(title));
        self.output.push('\n');
        self.row(&headers, &widths, &numeric);
        self.row(&widths.map(|width| "-".repeat(width)), &widths, &numeric);
        if rows.is_empty() {
            self.output.push_str("  (none)\n");
        } else {
            for row in rows {
                self.row(&row, &widths, &numeric);
            }
        }
    }

    /// Returns the report without a trailing newline.
    pub fn finish(mut self) -> String {
        if self.output.ends_with('\n') {
            self.output.pop();
        }
        self.output
    }

    fn row<const N: usize>(
        &mut self,
        cells: &[String; N],
        widths: &[usize; N],
        numeric: &[bool; N],
    ) {
        self.output.push_str("  ");
        for (column, ((cell, &width), &numeric)) in
            cells.iter().zip(widths).zip(numeric).enumerate()
        {
            if column > 0 {
                self.output.push_str("  ");
            }
            if numeric {
                self.output.push_str(&format!("{cell:>width$}"));
            } else if column + 1 == N {
                self.output.push_str(cell);
            } else {
                self.output.push_str(&format!("{cell:<width$}"));
            }
        }
        self.output.push('\n');
    }
}

/// A text cell or an unsigned number in a report table.
pub enum Value {
    /// An unsigned integer formatted with comma-separated decimal groups.
    Number(u64),
    /// Text with newlines, carriage returns, and tabs escaped.
    Text(String),
}

impl Value {
    /// Creates a text cell from a value's string representation.
    pub fn text(value: impl ToString) -> Self {
        Self::Text(value.to_string())
    }

    fn render(self) -> String {
        match self {
            Self::Number(value) => grouped_decimal(value),
            Self::Text(value) => single_line(&value),
        }
    }
}

fn single_line(value: &str) -> String {
    value
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

pub(crate) fn grouped_decimal(value: u64) -> String {
    let digits = value.to_string();
    let mut grouped = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            grouped.push(',');
        }
        grouped.push(digit);
    }
    grouped
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    #[rstest]
    #[case::zero(0, "0")]
    #[case::below_thousand(999, "999")]
    #[case::thousand(1_000, "1,000")]
    #[case::below_million(999_999, "999,999")]
    #[case::million(1_000_000, "1,000,000")]
    #[case::readable_example(1_300_233, "1,300,233")]
    #[case::maximum(u64::MAX, "18,446,744,073,709,551,615")]
    fn decimal_numbers_have_thousands_separators(#[case] value: u64, #[case] expected: &str) {
        assert_eq!(super::grouped_decimal(value), expected);
    }
}
