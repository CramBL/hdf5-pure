//! Deterministic inputs and filter cases for creation and pipeline measurements.

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

/// A pipeline or compressor control for allocation and throughput measurements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineCase {
    /// Deflate without shuffle.
    Deflate,
    /// Deflate followed by Fletcher32, without shuffle.
    DeflateFletcher32,
    /// Fletcher32 alone.
    Fletcher32,
    /// LZF without shuffle.
    Lzf,
    /// Byte shuffle alone.
    Shuffle,
    /// Byte shuffle followed by Deflate.
    ShuffleDeflate,
    /// Byte shuffle, Deflate, then Fletcher32.
    ShuffleDeflateFletcher32,
    /// Byte shuffle followed by Fletcher32.
    ShuffleFletcher32,
    /// Byte shuffle followed by LZF.
    ShuffleLzf,
}

impl PipelineCase {
    /// Returns the label used in measurement output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Deflate => "deflate",
            Self::DeflateFletcher32 => "deflate_fletcher32",
            Self::Fletcher32 => "fletcher32",
            Self::Lzf => "lzf",
            Self::Shuffle => "shuffle",
            Self::ShuffleDeflate => "shuffle_deflate",
            Self::ShuffleDeflateFletcher32 => "shuffle_deflate_fletcher32",
            Self::ShuffleFletcher32 => "shuffle_fletcher32",
            Self::ShuffleLzf => "shuffle_lzf",
        }
    }

    /// Returns the stages in forward execution order.
    pub fn filters(self) -> &'static [PipelineFilter] {
        use PipelineFilter::{Deflate, Fletcher32, Lzf, Shuffle};
        match self {
            Self::Deflate => &[Deflate],
            Self::DeflateFletcher32 => &[Deflate, Fletcher32],
            Self::Fletcher32 => &[Fletcher32],
            Self::Lzf => &[Lzf],
            Self::Shuffle => &[Shuffle],
            Self::ShuffleDeflate => &[Shuffle, Deflate],
            Self::ShuffleDeflateFletcher32 => &[Shuffle, Deflate, Fletcher32],
            Self::ShuffleFletcher32 => &[Shuffle, Fletcher32],
            Self::ShuffleLzf => &[Shuffle, Lzf],
        }
    }

    /// Returns the LZF or Deflate pipeline without shuffle.
    ///
    /// Checksum stages are preserved. Returns [`None`] for cases without a compressor preceded
    /// by shuffle.
    ///
    /// # Examples
    ///
    /// ```
    /// use test_util::filtered_chunks::PipelineCase;
    ///
    /// assert_eq!(
    ///     PipelineCase::ShuffleLzf.codec_control(),
    ///     Some(PipelineCase::Lzf),
    /// );
    /// ```
    pub fn codec_control(self) -> Option<Self> {
        match self {
            Self::ShuffleDeflate => Some(Self::Deflate),
            Self::ShuffleDeflateFletcher32 => Some(Self::DeflateFletcher32),
            Self::ShuffleLzf => Some(Self::Lzf),
            _ => None,
        }
    }
}

/// A byte stage in a measurement pipeline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineFilter {
    /// The zlib codec.
    Deflate,
    /// The Fletcher32 checksum.
    Fletcher32,
    /// The raw LZF codec.
    Lzf,
    /// Byte shuffling by element width.
    Shuffle,
}

/// The direction of one filter pipeline operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// Reverses the pipeline from last stage to first.
    Decode,
    /// Applies the stages in forward order.
    Encode,
}

impl Direction {
    /// Returns the label used in measurement output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Encode => "encode",
        }
    }
}

/// Returns `elements` four-byte little-endian values cycling from zero through 31.
pub fn pipeline_data(elements: usize) -> Vec<u8> {
    (0..elements)
        .flat_map(|index| u32::try_from(index % 32).unwrap().to_le_bytes())
        .collect()
}

/// Groups bytes of complete four-byte elements by byte position, preserving trailing bytes.
pub fn shuffled_pipeline_data(raw: &[u8]) -> Vec<u8> {
    let (elements, tail) = raw.as_chunks::<4>();
    (0..4)
        .flat_map(|byte| elements.iter().map(move |element| element[byte]))
        .chain(tail.iter().copied())
        .collect()
}

/// Filter combinations for the direct allocation and throughput workloads.
///
/// Compressor controls are selected through [`PipelineCase::codec_control`].
pub const PIPELINES: [PipelineCase; 6] = [
    PipelineCase::Shuffle,
    PipelineCase::Fletcher32,
    PipelineCase::ShuffleFletcher32,
    PipelineCase::ShuffleLzf,
    PipelineCase::ShuffleDeflate,
    PipelineCase::ShuffleDeflateFletcher32,
];

/// A deterministic input sequence for direct pipeline measurements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelinePattern {
    /// Four-byte values cycling from zero through 31.
    Cycling,
    /// Four-byte xorshift values generated from a fixed seed.
    Incompressible,
}

impl PipelinePattern {
    /// Returns four-byte little-endian elements from the selected sequence.
    ///
    /// # Examples
    ///
    /// ```
    /// use test_util::filtered_chunks::PipelinePattern;
    ///
    /// assert_eq!(PipelinePattern::Cycling.data(2), [0, 0, 0, 0, 1, 0, 0, 0]);
    /// assert_eq!(
    ///     PipelinePattern::Incompressible.data(1),
    ///     [0xa5, 0x5a, 0x98, 0x87],
    /// );
    /// ```
    pub fn data(self, elements: usize) -> Vec<u8> {
        match self {
            Self::Cycling => pipeline_data(elements),
            Self::Incompressible => {
                let mut state = 0x1234_5678_u32;
                (0..elements)
                    .flat_map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        state.to_le_bytes()
                    })
                    .collect()
            }
        }
    }

    /// Returns the label used in measurement output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cycling => "cycling",
            Self::Incompressible => "incompressible",
        }
    }
}

/// The scratch acquisition policy for one measured operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScratchMode {
    /// Acquires fresh scratch for each operation.
    Cold,
    /// Reuses scratch after two warmup operations.
    Reused,
}

impl ScratchMode {
    /// Returns the label used in measurement output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Reused => "reused",
        }
    }
}

/// Four-byte element counts around 4 KiB and 32 KiB, with a 64 KiB case.
pub const PIPELINE_ELEMENTS: [usize; 7] = [1023, 1024, 1025, 8191, 8192, 8193, 16_384];
