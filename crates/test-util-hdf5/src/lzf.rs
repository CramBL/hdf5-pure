//! LZF datasets shared by pure round trips and native interoperability tests.

use hdf5_pure::File;
use hdf5_pure::FileBuilder;
use hdf5_pure::FilterId;

/// Values for plain, shuffled, partial-edge, and incompressible LZF datasets.
///
/// The noise values come from the committed `h5py/lzf/u8_noise.raw.bin` vector.
/// [`Self::pure_bytes`] constructs a file with the current writer.
///
/// # Examples
///
/// ```
/// use hdf5_pure::File;
/// use test_util_hdf5::lzf::Fixture;
///
/// let fixture = Fixture::new();
/// let file = File::from_bytes(fixture.pure_bytes())?;
/// fixture.assert_pure(&file);
/// # Ok::<(), hdf5_pure::Error>(())
/// ```
pub struct Fixture {
    integers: Vec<i32>,
    shuffled: Vec<f64>,
    partial: Vec<i64>,
    noise: [u8; 4096],
}

impl Fixture {
    /// Constructs deterministic values and loads the committed noise vector.
    ///
    /// # Panics
    ///
    /// Panics if the noise vector cannot be read or does not contain 4096 bytes.
    pub fn new() -> Self {
        Self {
            integers: (0..1024).collect(),
            shuffled: (0..512).map(|value| f64::from(value).sqrt()).collect(),
            partial: (0..1000).map(|value| value * 7 - 500).collect(),
            noise: std::fs::read(test_util::data("h5py/lzf/u8_noise.raw.bin"))
                .unwrap()
                .try_into()
                .unwrap(),
        }
    }

    /// Encodes the datasets with LZF, adding shuffle before LZF for the floating-point values.
    ///
    /// # Panics
    ///
    /// Panics if the writer cannot encode the fixture.
    pub fn pure_bytes(&self) -> Vec<u8> {
        let Self {
            integers,
            shuffled,
            partial,
            noise,
        } = self;
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("plain_i32")
            .with_i32_data(integers)
            .with_chunks(&[256])
            .with_lzf();
        builder
            .create_dataset("shuffle_f64")
            .with_f64_data(shuffled)
            .with_chunks(&[128])
            .with_shuffle()
            .with_lzf();
        builder
            .create_dataset("multichunk_i64")
            .with_i64_data(partial)
            .with_chunks(&[128])
            .with_lzf();
        builder
            .create_dataset("incompressible_u8")
            .with_u8_data(noise)
            .with_chunks(&[1024])
            .with_lzf();
        builder.finish().unwrap()
    }

    /// Checks the fixture's dataset values and the shuffle/LZF pipeline.
    ///
    /// # Panics
    ///
    /// Panics if a dataset cannot be opened or read, or its values or pipeline differ.
    pub fn assert_pure(&self, file: &File) {
        assert_eq!(
            file.dataset("plain_i32").unwrap().read_i32().unwrap(),
            self.integers
        );
        assert_eq!(
            file.dataset("shuffle_f64").unwrap().read_f64().unwrap(),
            self.shuffled
        );
        assert_eq!(
            file.dataset("multichunk_i64").unwrap().read_i64().unwrap(),
            self.partial
        );
        assert_eq!(
            file.dataset("incompressible_u8")
                .unwrap()
                .read_u8()
                .unwrap(),
            self.noise
        );
        assert_eq!(
            file.dataset("shuffle_f64").unwrap().filters().unwrap(),
            [FilterId::Shuffle, FilterId::Lzf]
        );
    }

    /// Returns the values stored in `plain_i32`.
    pub fn integers(&self) -> &[i32] {
        &self.integers
    }

    /// Returns the values stored in `shuffle_f64`.
    pub fn shuffled(&self) -> &[f64] {
        &self.shuffled
    }

    /// Returns the values stored in `multichunk_i64`.
    pub fn partial(&self) -> &[i64] {
        &self.partial
    }

    /// Returns the committed bytes stored in `incompressible_u8`.
    pub fn noise(&self) -> &[u8] {
        &self.noise
    }

    /// Returns different bytes for the incompressible native rewrite.
    ///
    /// The result has the same length as the original noise vector.
    ///
    /// # Panics
    ///
    /// Panics if the transformed vector is unchanged.
    pub fn rewritten_noise(&self) -> Vec<u8> {
        let mut rewritten = self.noise.to_vec();
        rewritten.rotate_right(1);
        for value in &mut rewritten {
            *value ^= 0xa5;
        }
        assert_ne!(rewritten, self.noise);
        rewritten
    }
}

impl Default for Fixture {
    fn default() -> Self {
        Self::new()
    }
}
