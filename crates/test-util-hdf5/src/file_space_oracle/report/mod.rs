//! Text reports of paired free-space observations and comparison results.

use test_util::snapshot::text::{Report, Value};

use super::{CreationProperties, FreeSpaceOracleCheckpoint, PropertyValues};

impl FreeSpaceOracleCheckpoint {
    /// Formats paired creation properties, file sizes, sections, and free-space totals.
    ///
    /// Uses comma-separated decimal groups and zero-padded hexadecimal addresses and lengths.
    /// Marks missing sections and unequal pairs explicitly. Includes section-byte sums and the
    /// separate `H5Fget_freespace` total.
    pub fn render(&self) -> String {
        let Self {
            case,
            step,
            rust,
            c,
            c_reported_freespace,
            rust_logical_size,
            c_file_size,
            physical_file_size,
            base: _,
            properties,
        } = self;
        let mut report = Report::new();
        report.table(
            "checkpoint",
            ["measurement", "value"],
            [
                [Value::text("case"), Value::text(case)],
                [Value::text("checkpoint"), Value::text(step)],
            ],
        );
        properties.render(&mut report);
        report.metrics(
            "file sizes (absolute bytes)",
            "bytes",
            [
                ("hdf5-pure stored EOF", *rust_logical_size),
                ("libhdf5 H5Fget_filesize", *c_file_size),
                ("physical file", *physical_file_size),
            ],
        );
        self.render_sections(&mut report);
        report.table(
            "free-space totals",
            ["measurement", "hdf5-pure", "libhdf5"],
            [
                [
                    Value::text("section count"),
                    Value::Number(rust.sections.len() as u64),
                    Value::Number(c.sections.len() as u64),
                ],
                [
                    Value::text("section bytes"),
                    Value::Number(rust.bytes),
                    Value::Number(c.bytes),
                ],
                [
                    Value::text("H5Fget_freespace"),
                    Value::text("-"),
                    Value::Number(*c_reported_freespace),
                ],
                [
                    Value::text("reported - section bytes"),
                    Value::text("-"),
                    Value::Signed(i128::from(*c_reported_freespace) - i128::from(c.bytes)),
                ],
            ],
        );
        report.table(
            "comparison",
            ["measurement", "value"],
            [[
                Value::text("exact section map equal"),
                Value::text(rust.sections == c.sections),
            ]],
        );
        format!("{}\n\n", report.finish())
    }

    fn render_sections(&self, report: &mut Report) {
        let count = self.rust.sections.len().max(self.c.sections.len());
        let (max_address, max_length) = self
            .rust
            .sections
            .iter()
            .chain(&self.c.sections)
            .fold((0, 0), |(address, length), &(at, len)| {
                (address.max(at), length.max(len))
            });
        let address_digits = format!("{max_address:x}").len().max(8);
        let length_digits = format!("{max_length:x}").len().max(8);
        report.table(
            "free-space sections (stored addresses, byte lengths)",
            [
                "index",
                "hdf5-pure address",
                "hdf5-pure length",
                "libhdf5 address",
                "libhdf5 length",
                "match",
            ],
            (0..count).map(|index| {
                let rust = self.rust.sections.get(index);
                let c = self.c.sections.get(index);
                let [rust_address, rust_length] =
                    section_values(rust, address_digits, length_digits);
                let [c_address, c_length] = section_values(c, address_digits, length_digits);
                [
                    Value::Number(index as u64),
                    rust_address,
                    rust_length,
                    c_address,
                    c_length,
                    Value::text(if rust == c { "yes" } else { "NO" }),
                ]
            }),
        );
    }
}

impl CreationProperties {
    fn render(&self, report: &mut Report) {
        let Self { rust, c } = self;
        let rust = rust.values();
        let c = c.values();
        report.table(
            "file-space properties",
            ["property", "hdf5-pure", "libhdf5"],
            [
                "strategy",
                "persistence",
                "threshold (bytes)",
                "page size (bytes)",
                "base / userblock (bytes)",
            ]
            .into_iter()
            .zip(rust)
            .zip(c)
            .map(|((label, rust), c)| [Value::text(label), rust, c]),
        );
    }
}

impl PropertyValues {
    fn values(&self) -> [Value; 5] {
        let Self {
            strategy,
            persist,
            threshold,
            page_size,
            base,
        } = *self;
        [
            Value::text(strategy),
            Value::text(persist),
            Value::Number(threshold),
            page_size.map_or_else(|| Value::text("not stored"), Value::Number),
            Value::Number(base),
        ]
    }
}

fn section_values(
    section: Option<&(u64, u64)>,
    address_digits: usize,
    length_digits: usize,
) -> [Value; 2] {
    match section {
        Some(&(address, length)) => [
            Value::Hexadecimal {
                value: address,
                digits: address_digits,
            },
            Value::Hexadecimal {
                value: length,
                digits: length_digits,
            },
        ],
        None => [Value::text("<missing>"), Value::text("<missing>")],
    }
}

#[cfg(test)]
mod tests;
