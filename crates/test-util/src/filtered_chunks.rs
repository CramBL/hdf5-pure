//! Deterministic dataset inputs shared by filtered-creation measurements.

/// Dataset geometry, filter selection, and value generation for one measurement.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    /// The case label used in benchmark and allocation results.
    pub name: &'static str,
    /// Dataset dimensions in elements.
    pub shape: [u64; 2],
    /// Full chunk dimensions in elements, including edge padding.
    pub chunks: [u64; 2],
    /// The value stored in unwritten chunk slots.
    pub fill: u32,
    /// Whether LZF compression follows byte shuffle.
    pub lzf: bool,
    /// Whether to generate a random sequence with a fixed seed.
    pub random: bool,
}

impl Workload {
    /// Returns repeated values from 0 through 31, or a random sequence with a fixed seed.
    ///
    /// The number of values is the product of [`shape`](Self::shape).
    ///
    /// # Panics
    ///
    /// Panics if the element count overflows with overflow checks enabled.
    pub fn data(self) -> Vec<u32> {
        let mut state = 0x1234_5678_u32;
        (0..self.shape.iter().product())
            .map(|index| {
                if self.random {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    state
                } else {
                    u32::try_from(index % 32).unwrap()
                }
            })
            .collect()
    }
}

/// Cases covering small chunks, incompressible data, padded edges, and full-width slabs.
pub const WORKLOADS: [Workload; 5] = [
    Workload {
        name: "small_shuffle",
        shape: [1_024, 1_024],
        chunks: [8, 16],
        fill: 0,
        lzf: false,
        random: false,
    },
    Workload {
        name: "small_lzf",
        shape: [1_024, 1_024],
        chunks: [8, 16],
        fill: 0,
        lzf: true,
        random: false,
    },
    Workload {
        name: "incompressible_lzf",
        shape: [2_048, 4_096],
        chunks: [64, 128],
        fill: 0,
        lzf: true,
        random: true,
    },
    Workload {
        name: "edges_custom_fill",
        shape: [1_025, 1_031],
        chunks: [32, 64],
        fill: 0x1234_abcd,
        lzf: true,
        random: false,
    },
    Workload {
        name: "full_width_shuffle",
        shape: [1_024, 1_024],
        chunks: [16, 1_024],
        fill: 0,
        lzf: false,
        random: true,
    },
];
