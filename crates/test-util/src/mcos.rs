//! The MCOS subsystem a MAT v7.3 file stores MATLAB objects in: the `FileWrapper__` metadata
//! blob, and the metadata an object's own variable holds.
//!
//! MathWorks documents neither. The layout is the one the `matio`, `MatFileHandler` and
//! `foreverallama` parsers read.

/// The bytes of a `FileWrapper__` metadata blob, cell 0 of `#subsystem#/MCOS`.
///
/// A caller gives only the one-based entries. The reserved leading class entry, object entry
/// and empty property block 0 of each segment are written for it.
#[derive(Clone, Debug, Default)]
pub struct FileWrapper {
    names: Vec<String>,
    classes: Vec<(u32, u32)>,
    objects: Vec<(u32, u32, u32)>,
    saveobj_blocks: Vec<Vec<(u32, u32, u32)>>,
    normal_blocks: Vec<Vec<(u32, u32, u32)>>,
}

impl FileWrapper {
    /// A blob whose name table holds `names`, which every other entry indexes from one.
    pub fn new(names: &[&str]) -> Self {
        Self {
            names: names.iter().map(|name| (*name).to_owned()).collect(),
            ..Self::default()
        }
    }

    /// A class, by the indices of its namespace (zero for none) and its name.
    pub fn class(mut self, namespace: u32, name: u32) -> Self {
        self.classes.push((namespace, name));
        self
    }

    /// An object of class `class` whose properties are the `saveobj` block and the normal block
    /// of those indices, zero for none.
    pub fn object(mut self, class: u32, saveobj_block: u32, normal_block: u32) -> Self {
        self.objects.push((class, saveobj_block, normal_block));
        self
    }

    /// A property block of the segment `saveobj` properties are stored in, as `(name index,
    /// field type, value)` triples.
    pub fn saveobj_block(mut self, properties: &[(u32, u32, u32)]) -> Self {
        self.saveobj_blocks.push(properties.to_vec());
        self
    }

    /// A property block of the segment ordinary properties are stored in, as `(name index,
    /// field type, value)` triples.
    pub fn normal_block(mut self, properties: &[(u32, u32, u32)]) -> Self {
        self.normal_blocks.push(properties.to_vec());
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let Self {
            names,
            classes,
            objects,
            saveobj_blocks,
            normal_blocks,
        } = self;

        let mut name_table = Vec::new();
        for name in names {
            name_table.extend_from_slice(name.as_bytes());
            name_table.push(0);
        }
        name_table.resize(name_table.len().next_multiple_of(8), 0);

        let class_table = words(
            &std::iter::once([0; 4])
                .chain(
                    classes
                        .iter()
                        .map(|&(namespace, name)| [namespace, name, 0, 0]),
                )
                .flatten()
                .collect::<Vec<_>>(),
        );
        let object_table = words(
            &std::iter::once([0; 6])
                .chain(
                    objects
                        .iter()
                        .map(|&(class, saveobj, normal)| [class, 0, 0, saveobj, normal, 0]),
                )
                .flatten()
                .collect::<Vec<_>>(),
        );
        let saveobj_segment = segment(saveobj_blocks);
        let normal_segment = segment(normal_blocks);
        let dynamic_properties = words(&property_block(&[]));

        // The regions in the order they follow the header, each offset naming where the next
        // one starts, and the last three the end of the dynamic-property table.
        let regions = [
            &class_table,
            &saveobj_segment,
            &object_table,
            &normal_segment,
            &dynamic_properties,
        ];
        let mut offsets = [0u32; 8];
        offsets[0] = u32::try_from(HEADER + name_table.len()).expect("a blob under 4 GiB");
        for (at, region) in regions.iter().enumerate() {
            let end = offsets[at] + u32::try_from(region.len()).expect("a blob under 4 GiB");
            offsets[at + 1..].fill(end);
        }

        let mut blob = VERSION.to_le_bytes().to_vec();
        blob.extend_from_slice(
            &u32::try_from(names.len())
                .expect("fewer than 2^32 names")
                .to_le_bytes(),
        );
        blob.extend_from_slice(&words(&offsets));
        blob.extend_from_slice(&name_table);
        for region in regions {
            blob.extend_from_slice(region);
        }
        blob
    }
}

/// The metadata the variable of one scalar object holds: the magic number, two dimensions of
/// one, and the object and class it refers to.
pub fn object_metadata(object: u32, class: u32) -> Vec<u32> {
    vec![MAGIC, 2, 1, 1, object, class]
}

/// A property block: its property count, then a `(name index, field type, value)` triple per
/// property, padded to a whole number of eight-byte units.
pub fn property_block(properties: &[(u32, u32, u32)]) -> Vec<u32> {
    let mut block = vec![u32::try_from(properties.len()).expect("fewer than 2^32 properties")];
    for &(name, field_type, value) in properties {
        block.extend_from_slice(&[name, field_type, value]);
    }
    block.resize(block.len().next_multiple_of(2), 0);
    block
}

// A segment of property blocks, led by the reserved empty block 0.
fn segment(blocks: &[Vec<(u32, u32, u32)>]) -> Vec<u8> {
    words(
        &std::iter::once(property_block(&[]))
            .chain(blocks.iter().map(|block| property_block(block)))
            .flatten()
            .collect::<Vec<_>>(),
    )
}

fn words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// The first word of an object's metadata.
pub const MAGIC: u32 = 0xDD00_0000;

/// A property whose value is the index of the MCOS cell holding it, counted from the first cell
/// after the blob and the canonical empty.
pub const HEAP: u32 = 1;

/// A property whose value is stored in the triple itself.
pub const INLINE: u32 = 2;

/// The version the blob declares.
const VERSION: u32 = 4;

/// The version, the name count and the eight region offsets.
const HEADER: usize = 40;

#[cfg(test)]
mod tests {
    use crate::bytes;
    use crate::mcos::{self, FileWrapper};

    #[test]
    fn the_offsets_name_where_each_region_starts() {
        let blob = FileWrapper::new(&["datetime", "data"])
            .class(0, 1)
            .object(1, 0, 1)
            .normal_block(&[(2, mcos::HEAP, 0)])
            .build();

        // Two names pad to sixteen bytes, and the regions after them are 32, 8, 48, 24 and 8
        // bytes long.
        let offsets: Vec<u32> = (0..8).map(|i| bytes::u32_at(&blob, 8 + 4 * i)).collect();
        assert_eq!(offsets, vec![56, 88, 96, 144, 168, 176, 176, 176]);
        assert_eq!(blob.len(), 176);
    }
}
