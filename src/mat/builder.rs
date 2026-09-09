//! Mid-level builder API for MAT v7.3 files.
//!
//! [`MatBuilder`] wraps [`crate::FileBuilder`] and applies MATLAB conventions:
//! `MATLAB_class` attributes, `MATLAB_int_decode` flags, `MATLAB_empty`
//! markers, the lazy `#refs#` group for object references, and the
//! `#subsystem#/MCOS` opaque-class subsystem for `string` objects.
//!
//! Callers hand in MATLAB-shape `dims` (e.g. `[N, 1]` for a column vector);
//! the builder transposes to HDF5 storage shape internally. Numeric data is
//! taken by `&[T]` so callers don't need to allocate per dataset.
//!
//! Nested structs and cells use closures so the borrow checker can keep the
//! call tree straight:
//!
//! ```ignore
//! let mut mb = MatBuilder::new(Options::default());
//! mb.struct_("payload", |s| {
//!     s.write_scalar_f64("answer", 42.0)?;
//!     s.cell("entries", &[2, 1], |c| {
//!         c.push_scalar_f64(1.0)?;
//!         c.push_string("hi")?;
//!         Ok(())
//!     })?;
//!     Ok(())
//! })?;
//! let bytes = mb.finish()?;
//! ```
//!
//! The builder allocates exactly one `#refs#/ref_{id:016x}` per cell element
//! and one per string-object saveobj payload; both groups are created lazily
//! on first use. No intermediate value tree is allocated.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::file_writer::AttrValue;
use crate::mat::class::MatClass;
use crate::mat::dims::{
    STORAGE_DIMS_BUF_LEN, matrix_dims, storage_dims_u64, storage_dims_u64_into, vector_dims,
};
use crate::mat::error::MatError;
use crate::mat::identifier::{dedupe_name, is_valid_name, sanitize_name};
use crate::mat::options::{Compression, EmptyMarkerEncoding, InvalidNamePolicy, Options};
use crate::mat::producer::{
    BlockElement, DataProducer, ProducerChunks, plan_blocking, total_elements,
};
use crate::mat::string_object;
use crate::mat::userblock::{self, USERBLOCK_SIZE};
use crate::mat::utf16;
use crate::type_builders::{ComplexComponent, DatasetBuilder, GroupBuilder};
use crate::writer::FileBuilder;

const REFS_GROUP: &str = "#refs#";
const SUBSYSTEM_GROUP: &str = "#subsystem#";

/// The `H5PATH` attribute MATLAB stamps on an object interned under `#refs#`:
/// the object's own absolute path.
///
/// Measured across the MATLAB-authored fixtures in `tests/data/matlab`:
/// every object directly under `#refs#` carries it — 519 of them — with two
/// exceptions that hold in all eight files, the `canonical empty` placeholder
/// and `#subsystem#/MCOS`. Set before the `MATLAB_*` attributes, which is the
/// order MATLAB writes them in.
///
/// MATLAB also stamps objects *deeper* than that, with a value that is not
/// their path: a field of the struct at `#refs#/j` gets `/#refs#/jj`, its
/// parent's value with the parent's own name appended again, which names an
/// object the file does not contain. That artifact is not reproduced — nothing
/// below `#refs#`'s immediate children carries the attribute here.
pub(crate) fn refs_h5path(ref_name: &str) -> AttrValue {
    AttrValue::AsciiString(format!("/{REFS_GROUP}/{ref_name}"))
}

/// Whether an object being written carries [`refs_h5path`]. Shared by both MAT
/// emitters, so the rule has one definition rather than one per writer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefsH5Path {
    /// Stamp the object's own path. Every `#refs#` member but one.
    Own,
    /// Write no `H5PATH`: the object is not a `#refs#` member, or it is the
    /// `canonical empty` placeholder, which MATLAB leaves unstamped.
    Omit,
}

/// Top-level MAT v7.3 writer.
pub struct MatBuilder {
    file: FileBuilder,
    options: Options,
    /// Lazy `#refs#` group, created on first use.
    refs: Option<GroupBuilder>,
    /// Monotonic counter for `ref_{id:016x}` names.
    next_ref_id: u64,
    /// Saveobj payload paths registered by `write_string_object`. Each entry
    /// becomes one MCOS reference.
    string_object_payload_paths: Vec<String>,
    /// Names already used at the file root.
    root_used_names: HashSet<String>,
    /// Stack of open struct groups. The deepest is the current write target
    /// when [`next_target`](Self::next_target) is `None`.
    open_structs: Vec<OpenStruct>,
    /// Single-use override for the next `write_*` / `struct_` / `cell` call.
    /// Set by [`CellWriter`] before pushing a cell element; consumed by the
    /// next write to redirect the dataset/group into `#refs#`.
    next_target: Option<NextTarget>,
    /// Where a [`DataProducer`]'s own error lands. The writer's provider seam
    /// speaks `FormatError` and cannot carry a `MatError`, so a producer failing
    /// mid-write stashes the real error here and the finalizers swap it back for
    /// the placeholder the writer surfaced. Empty unless a producer failed.
    producer_error: Arc<Mutex<Option<MatError>>>,
}

struct OpenStruct {
    group: GroupBuilder,
    used_names: HashSet<String>,
    fields: Vec<String>,
    parent: ParentKind,
}

enum ParentKind {
    /// Closing this struct attaches the finished group to the file root.
    Root,
    /// Closing this struct attaches the finished group to the open-struct at
    /// the given index of `open_structs` (after this one is popped).
    StructAt(usize),
    /// Closing this struct attaches the finished group to the `#refs#` group
    /// (using the group's own name, which is `ref_NNNN`).
    Refs,
}

/// Single-use override consumed by the next write/open.
struct NextTarget {
    /// The ref name (e.g. `"ref_0000000000000000"`). The dataset/group will
    /// be placed in `#refs#` with this name.
    ref_name: String,
}

impl MatBuilder {
    /// Construct a new MAT writer.
    pub fn new(options: Options) -> Self {
        let mut file = FileBuilder::new();
        file.with_userblock(USERBLOCK_SIZE);
        // The upper bound is the caller's; the lower one is always `Earliest`,
        // since a MAT file has no reason to demand a *newer* reader than its
        // content forces. See `Options::libver` for why the default is 1.8.
        file.with_libver_bounds(crate::LibVer::Earliest, options.libver);
        Self {
            file,
            options,
            refs: None,
            next_ref_id: 0,
            string_object_payload_paths: Vec::new(),
            root_used_names: HashSet::new(),
            open_structs: Vec::new(),
            next_target: None,
            producer_error: Arc::new(Mutex::new(None)),
        }
    }

    /// Borrow the configured options.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Allocate a fresh `ref_{:016x}` name. Does not write anything.
    pub fn alloc_ref_name(&mut self) -> String {
        let name = format!("ref_{:016x}", self.next_ref_id);
        self.next_ref_id += 1;
        name
    }

    /// Get-or-create the `#refs#` group on the underlying file.
    fn refs_mut(&mut self) -> &mut GroupBuilder {
        if self.refs.is_none() {
            self.refs = Some(self.file.create_group(REFS_GROUP));
        }
        self.refs.as_mut().unwrap()
    }

    fn normalize_name(
        raw: &str,
        used: &mut HashSet<String>,
        policy: InvalidNamePolicy,
    ) -> Result<String, MatError> {
        let candidate = match policy {
            InvalidNamePolicy::Error => {
                if !is_valid_name(raw) {
                    return Err(MatError::Custom(format!("invalid MATLAB name: {raw}")));
                }
                raw.to_owned()
            }
            InvalidNamePolicy::Sanitize => sanitize_name(raw),
        };
        let unique = dedupe_name(candidate, used);
        used.insert(unique.clone());
        Ok(unique)
    }

    /// Resolve a name in the current write scope and return the (resolved
    /// name, write target) pair to use for the next dataset/group.
    fn resolve_target(&mut self, raw_name: &str) -> Result<TargetForName, MatError> {
        if let Some(t) = self.next_target.take() {
            // Cell-element / explicit-ref target. The user-supplied name is
            // ignored; the ref name is fixed.
            return Ok(TargetForName::Ref(t.ref_name));
        }
        if let Some(top) = self.open_structs.last_mut() {
            let name = Self::normalize_name(
                raw_name,
                &mut top.used_names,
                self.options.invalid_name_policy,
            )?;
            top.fields.push(name.clone());
            return Ok(TargetForName::Struct {
                name,
                index: self.open_structs.len() - 1,
            });
        }
        let name = Self::normalize_name(
            raw_name,
            &mut self.root_used_names,
            self.options.invalid_name_policy,
        )?;
        Ok(TargetForName::Root(name))
    }

    /// Mutably borrow a `DatasetBuilder` at the resolved target.
    fn dataset_at_target<'a>(&'a mut self, target: &'a TargetForName) -> &'a mut DatasetBuilder {
        match target {
            TargetForName::Root(name) => self.file.create_dataset(name),
            TargetForName::Struct { name, index } => {
                self.open_structs[*index].group.create_dataset(name)
            }
            TargetForName::Ref(ref_name) => {
                let ds = self.refs_mut().create_dataset(ref_name);
                ds.set_attr("H5PATH", refs_h5path(ref_name));
                ds
            }
        }
    }

    // -- public scalar writes ---------------------------------------------

    pub fn write_scalar_logical(&mut self, name: &str, value: bool) -> Result<&mut Self, MatError> {
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_u8_data(&[u8::from(value)]).with_shape(&[1, 1]);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Logical.as_str().into()),
        );
        ds.set_attr("MATLAB_int_decode", AttrValue::I32(1));
        Ok(self)
    }

    pub fn write_scalar_f64(&mut self, name: &str, value: f64) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Double, |ds| {
            ds.with_f64_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_f32(&mut self, name: &str, value: f32) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Single, |ds| {
            ds.with_f32_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_i8(&mut self, name: &str, value: i8) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Int8, |ds| {
            ds.with_i8_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_i16(&mut self, name: &str, value: i16) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Int16, |ds| {
            ds.with_i16_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_i32(&mut self, name: &str, value: i32) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Int32, |ds| {
            ds.with_i32_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_i64(&mut self, name: &str, value: i64) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::Int64, |ds| {
            ds.with_i64_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_u8(&mut self, name: &str, value: u8) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::UInt8, |ds| {
            ds.with_u8_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_u16(&mut self, name: &str, value: u16) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::UInt16, |ds| {
            ds.with_u16_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_u32(&mut self, name: &str, value: u32) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::UInt32, |ds| {
            ds.with_u32_data(&[value]).with_shape(&[1, 1]);
        })
    }
    pub fn write_scalar_u64(&mut self, name: &str, value: u64) -> Result<&mut Self, MatError> {
        self.write_scalar_inner(name, MatClass::UInt64, |ds| {
            ds.with_u64_data(&[value]).with_shape(&[1, 1]);
        })
    }

    fn write_scalar_inner<F>(
        &mut self,
        name: &str,
        class: MatClass,
        apply: F,
    ) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut DatasetBuilder),
    {
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        apply(ds);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(class.as_str().into()),
        );
        Ok(self)
    }

    // -- numeric arrays ----------------------------------------------------

    pub fn write_f64(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[f64],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Double,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_f64_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_f32(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[f32],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Single,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_f32_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_i8(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[i8],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Int8,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_i8_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_i16(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[i16],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Int16,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_i16_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_i32(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[i32],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Int32,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_i32_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_i64(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[i64],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::Int64,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_i64_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_u8(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u8],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::UInt8,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u8_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_u16(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u16],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::UInt16,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u16_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_u32(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u32],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::UInt32,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u32_data(data).with_shape(shape);
            },
        )
    }
    pub fn write_u64(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u64],
    ) -> Result<&mut Self, MatError> {
        self.write_array_inner(
            name,
            MatClass::UInt64,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u64_data(data).with_shape(shape);
            },
        )
    }

    /// Write a logical array (MATLAB `logical`, stored as `uint8`).
    pub fn write_logical(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        data: &[u8],
    ) -> Result<&mut Self, MatError> {
        if data.is_empty() {
            return self.write_empty(name, MatClass::Logical, matlab_dims);
        }
        self.write_array_inner(
            name,
            MatClass::Logical,
            matlab_dims,
            data.len(),
            |ds, shape| {
                ds.with_u8_data(data).with_shape(shape);
                ds.set_attr("MATLAB_int_decode", AttrValue::I32(1));
            },
        )
    }

    /// Write a `char` UTF-16 string (MATLAB `char`). Uses `[N, 1]` MATLAB
    /// shape (a row vector when read back, since HDF5 storage transposes).
    pub fn write_char(&mut self, name: &str, value: &str) -> Result<&mut Self, MatError> {
        let units = utf16::encode_utf16(value);
        if units.is_empty() {
            return self.write_empty(name, MatClass::Char, &[0, 0]);
        }
        let n = units.len() as u64;
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_u16_data(&units).with_shape(&[n, 1]);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Char.as_str().into()),
        );
        ds.set_attr("MATLAB_int_decode", AttrValue::I32(2));
        Ok(self)
    }

    /// Write a complex array of the given component type.
    ///
    /// The dataset is a `{real, imag}` compound of `T` and carries the
    /// *component* class in `MATLAB_class` — `"int16"` for a complex `i16`
    /// array — which is how MATLAB distinguishes `complex(int16(re),
    /// int16(im))` from a complex `double`. No width is ever substituted for
    /// another: the class the file reports is the one the caller asked for.
    fn write_complex_of<T: ComplexComponent>(
        &mut self,
        name: &str,
        class: MatClass,
        matlab_dims: &[usize],
        data: &[(T, T)],
    ) -> Result<&mut Self, MatError> {
        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        let compression = self.options.compression;
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_complex_data(data).with_shape(storage);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(class.as_str().into()),
        );
        if !data.is_empty() {
            apply_deflate(ds, compression);
        }
        Ok(self)
    }

    /// Stage a numeric array whose bytes are produced at write time, so the
    /// dataset is never fully resident.
    ///
    /// `producer` is called once per block, in ascending index order, while the
    /// file is being emitted — not while it is being laid out, which works from
    /// the shape alone. Combined with [`finish_to`](Self::finish_to), which does
    /// not hold the assembled file either, this writes a `.mat` in one block plus
    /// the file's metadata, whatever its size.
    ///
    /// Each call hands the producer a [`Block`](crate::mat::Block) carrying its
    /// index, its first element's position, and how many elements to write, so a
    /// producer needs nothing from this call's return value. Block `i` is a
    /// contiguous run of MATLAB's linear element order, continuing where block
    /// `i - 1` stopped — see the [module docs](crate::mat::producer) for what
    /// that means for a shape like `[channels, samples]`.
    ///
    /// [`Blocking::plan`](crate::mat::Blocking::plan) computes the same split from the shape alone, ahead of
    /// staging, for a caller that wants to size a buffer or report progress.
    ///
    /// The dataset is byte-for-byte the one [`write_f64`](Self::write_f64) and its
    /// siblings produce for the same content; only the peak memory differs.
    ///
    /// The element type names the MATLAB class: `write_blocks::<i16>` writes an
    /// `int16` array, `write_blocks::<(i16, i16)>` a complex one, and
    /// `write_blocks::<bool>` a `logical`.
    ///
    /// An empty shape writes the same MATLAB empty marker the other writers do
    /// and never calls the producer.
    ///
    /// # Errors
    ///
    /// Refused up front if this builder is configured for compression: a block's
    /// on-disk size has to be known before any byte is written, and a compressed
    /// one is not knowable without compressing it. Refused at write time with
    /// [`MatError::BlockSizeMismatch`] if the producer writes the wrong number of
    /// bytes for a block, and a producer's own error is surfaced verbatim.
    ///
    /// # Example
    ///
    /// A ramp of one million `f64`, written without ever holding more than a
    /// block of it. The producer computes its bytes from the block index; a real
    /// one would read them from a capture buffer or a socket.
    ///
    /// ```
    /// use hdf5_pure::mat::{Block, DataProducer, MatBuilder, MatError, Options};
    ///
    /// // Nothing to configure: every block says how much to write and where it
    /// // starts, so the producer holds no state at all.
    /// struct Ramp;
    ///
    /// impl DataProducer for Ramp {
    ///     fn block_bytes(&self, block: Block, out: &mut Vec<u8>) -> Result<(), MatError> {
    ///         for i in 0..block.elements {
    ///             let value = (block.first_element + i) as f64;
    ///             out.extend_from_slice(&value.to_le_bytes());
    ///         }
    ///         Ok(())
    ///     }
    /// }
    ///
    /// // `[channels, samples]`: blocks then run forward through time.
    /// let dims = [4, 250_000];
    /// let mut mb = MatBuilder::new(Options::default());
    /// mb.write_blocks::<f64>("samples", &dims, Box::new(Ramp))
    ///     .unwrap()
    ///     .write_scalar_f64("sample_rate", 48_000.0)
    ///     .unwrap();
    ///
    /// let mut sink: Vec<u8> = Vec::new();
    /// mb.finish_to(&mut sink).unwrap();
    /// ```
    pub fn write_blocks<T: BlockElement>(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        producer: Box<dyn DataProducer>,
    ) -> Result<&mut Self, MatError> {
        if self.options.compression != Compression::None {
            return Err(MatError::CompressionUnsupportedForBlocks);
        }

        let blocking = plan_blocking(total_elements(matlab_dims), T::ELEMENT_SIZE)?;
        if blocking.block_count == 0 {
            return self.write_empty(name, T::CLASS, matlab_dims);
        }

        // `storage_dims_u64`, not the stack-buffer form the slice writers use:
        // this one takes its shape from the caller with no accompanying slice to
        // bound its rank, and the stack form panics past
        // `STORAGE_DIMS_BUF_LEN` dimensions.
        let storage = storage_dims_u64(matlab_dims);
        // Cloned before the builder is borrowed: the adapter needs the stash and
        // `dataset_at_target` holds `self` for the rest of the function.
        let error = Arc::clone(&self.producer_error);
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_produced_data(
            T::datatype(),
            &storage,
            blocking.total_len(),
            blocking.block_elements * T::ELEMENT_SIZE as u64,
            Box::new(ProducerChunks {
                producer,
                blocking,
                error,
            }),
        );
        // Decode flag before class, which is the order `write_array_inner` ends
        // up in: it runs the per-type closure — where `write_logical` sets the
        // flag — and only then sets the class. Attribute messages are emitted in
        // insertion order, so the reverse would give a produced `logical` a
        // different object header from a materialized one.
        if let Some(decode) = T::INT_DECODE {
            ds.set_attr("MATLAB_int_decode", AttrValue::I32(decode));
        }
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(T::CLASS.as_str().into()),
        );
        Ok(self)
    }

    /// Write a MATLAB `string` object. Allocates a `#refs#` payload entry and
    /// registers the object in the MCOS subsystem.
    pub fn write_string_object(
        &mut self,
        name: &str,
        values: &[String],
        matlab_dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        // 1. Encode payload, write to #refs# under a fresh ref.
        let payload = string_object::encode_string_saveobj_payload(values, matlab_dims)?;
        let payload_ref = self.alloc_ref_name();
        let payload_path = format!("{REFS_GROUP}/{payload_ref}");
        let payload_shape: [u64; 2] = [1, payload.len() as u64];
        {
            let refs = self.refs_mut();
            let ds = refs.create_dataset(&payload_ref);
            ds.set_attr("H5PATH", refs_h5path(&payload_ref));
            ds.with_u64_data(&payload).with_shape(&payload_shape);
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MatClass::UInt64.as_str().into()),
            );
        }

        // 2. Register the payload path; the resulting object id is 1-based.
        self.string_object_payload_paths.push(payload_path);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "1-based object id written into a 4-byte on-disk field; a MAT file never registers more than u32::MAX string objects"
        )]
        let object_id = self.string_object_payload_paths.len() as u32;

        // 3. Emit the parent metadata dataset at the current scope.
        let metadata = string_object::create_string_object_metadata(object_id);
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        ds.with_u32_data(&metadata)
            .with_shape(&[1, metadata.len() as u64]);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(string_object::MATLAB_CLASS_STRING.into()),
        );
        ds.set_attr(
            "MATLAB_object_decode",
            AttrValue::I32(string_object::MATLAB_OBJECT_DECODE_OPAQUE),
        );
        Ok(self)
    }

    /// Write an empty marker for the given class. `matlab_dims` is the
    /// MATLAB-shape (e.g. `[0, 0]`).
    ///
    /// An empty marker carries `MATLAB_class` and `MATLAB_empty` and nothing
    /// else. In particular it carries no `MATLAB_int_decode`, which says how to
    /// read stored integers back as `char` or `logical` values and so has nothing
    /// to describe here: the payload is a `uint64` dimension vector, not data of
    /// the marked class. Of the 352 empty datasets in the MATLAB-authored
    /// fixtures under `tests/data/matlab`, not one carries it, for any
    /// class.
    pub fn write_empty(
        &mut self,
        name: &str,
        class: MatClass,
        matlab_dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        let target = self.resolve_target(name)?;
        let encoding = self.options.empty_marker_encoding;
        let ds = self.dataset_at_target(&target);
        emit_empty_storage(ds, encoding, class, matlab_dims);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(class.as_str().into()),
        );
        ds.set_attr("MATLAB_empty", AttrValue::U32(1));
        Ok(self)
    }

    /// Write a MATLAB `struct([])` empty marker.
    pub fn write_empty_struct_array(&mut self, name: &str) -> Result<&mut Self, MatError> {
        self.write_empty(name, MatClass::Struct, &[0, 0])
    }

    fn write_array_inner<F>(
        &mut self,
        name: &str,
        class: MatClass,
        matlab_dims: &[usize],
        data_len: usize,
        apply: F,
    ) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut DatasetBuilder, &[u64]),
    {
        if data_len == 0 {
            return self.write_empty(name, class, matlab_dims);
        }
        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        let compression = self.options.compression;
        let target = self.resolve_target(name)?;
        let ds = self.dataset_at_target(&target);
        apply(ds, storage);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(class.as_str().into()),
        );
        apply_deflate(ds, compression);
        Ok(self)
    }

    // -- struct nesting ----------------------------------------------------

    /// Open a struct group at the current scope. Use [`StructWriter`] inside
    /// the closure to write fields.
    pub fn struct_<F>(&mut self, name: &str, fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut StructWriter) -> Result<(), MatError>,
    {
        // Resolve target and parent kind.
        let (group_name, parent) = match self.next_target.take() {
            Some(t) => (t.ref_name.clone(), ParentKind::Refs),
            None => {
                if let Some(top) = self.open_structs.last_mut() {
                    let resolved = Self::normalize_name(
                        name,
                        &mut top.used_names,
                        self.options.invalid_name_policy,
                    )?;
                    top.fields.push(resolved.clone());
                    (resolved, ParentKind::StructAt(self.open_structs.len() - 1))
                } else {
                    let resolved = Self::normalize_name(
                        name,
                        &mut self.root_used_names,
                        self.options.invalid_name_policy,
                    )?;
                    (resolved, ParentKind::Root)
                }
            }
        };

        let mut group = GroupBuilder::new(&group_name);
        if matches!(parent, ParentKind::Refs) {
            // Before the struct's own attributes, which `close_struct` sets:
            // MATLAB's order on a `#refs#` struct group is H5PATH, then
            // MATLAB_class, then MATLAB_fields.
            group.set_attr("H5PATH", refs_h5path(&group_name));
        }
        self.open_structs.push(OpenStruct {
            group,
            used_names: HashSet::new(),
            fields: Vec::new(),
            parent,
        });

        let res = {
            let mut sw = StructWriter { mb: self };
            fill(&mut sw)
        };

        // Always close even if fill errored, to keep the stack consistent.
        let close_res = self.close_struct();
        res?;
        close_res?;
        Ok(self)
    }

    fn close_struct(&mut self) -> Result<(), MatError> {
        let mut s = self
            .open_structs
            .pop()
            .ok_or_else(|| MatError::Custom("close_struct called with no open struct".into()))?;
        s.group.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Struct.as_str().into()),
        );
        s.group
            .set_attr("MATLAB_fields", AttrValue::VarLenAsciiCharArray(s.fields));
        let finished = s.group.finish();
        match s.parent {
            ParentKind::Root => self.file.add_group(finished),
            ParentKind::StructAt(idx) => self.open_structs[idx].group.add_group(finished),
            ParentKind::Refs => self.refs_mut().add_group(finished),
        }
        Ok(())
    }

    // -- cell arrays -------------------------------------------------------

    /// Write a cell array. The closure pushes elements; the parent dataset is
    /// emitted with object references after the closure returns.
    pub fn cell<F>(
        &mut self,
        name: &str,
        matlab_dims: &[usize],
        fill: F,
    ) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut CellWriter) -> Result<(), MatError>,
    {
        // Resolve where the cell parent dataset goes BEFORE running fill (so
        // pushes from fill don't mutate the resolved scope).
        let parent_target = self.resolve_target(name)?;

        let mut paths: Vec<String> = Vec::new();
        let res = {
            let mut cw = CellWriter {
                mb: self,
                paths: &mut paths,
            };
            fill(&mut cw)
        };
        res?;

        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        if paths.is_empty() {
            // Empty cell: emit a class=cell empty marker at the resolved
            // target. Reuse the empty-marker emit path.
            let encoding = self.options.empty_marker_encoding;
            let mut dim_buf = [0u64; STORAGE_DIMS_BUF_LEN];
            let ds = self.dataset_at_target(&parent_target);
            match encoding {
                EmptyMarkerEncoding::ZeroElement => {
                    emit_zero_element(ds, MatClass::Cell, storage);
                }
                EmptyMarkerEncoding::DataAsDims => {
                    if matlab_dims.len() > STORAGE_DIMS_BUF_LEN {
                        let dim_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();
                        ds.with_u64_data(&dim_data)
                            .with_shape(&[dim_data.len() as u64]);
                    } else {
                        let n = matlab_dims.len();
                        for (slot, &d) in dim_buf[..n].iter_mut().zip(matlab_dims) {
                            *slot = d as u64;
                        }
                        ds.with_u64_data(&dim_buf[..n]).with_shape(&[n as u64]);
                    }
                }
            }
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MatClass::Cell.as_str().into()),
            );
            ds.set_attr("MATLAB_empty", AttrValue::U32(1));
            return Ok(self);
        }

        let path_strs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let ds = self.dataset_at_target(&parent_target);
        ds.with_path_references(&path_strs).with_shape(storage);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(MatClass::Cell.as_str().into()),
        );
        Ok(self)
    }

    // -- helpers -----------------------------------------------------------

    /// MATLAB shape for a 1-D vector of length `len`, using the configured
    /// 1-D mode. Returns a stack-allocated 2-element array.
    #[inline]
    pub fn vector_dims(&self, len: usize) -> [usize; 2] {
        vector_dims(len, self.options.one_dimensional_mode)
    }

    /// MATLAB shape for a value with the given multi-dimensional `extents`.
    /// Allocates only when `extents.len() > 2`; the 0/1/2-D cases construct
    /// a `Vec<usize>` with capacity 2.
    #[inline]
    pub fn matrix_dims(&self, extents: &[usize]) -> Vec<usize> {
        matrix_dims(extents, self.options.one_dimensional_mode)
    }

    // -- finalize ----------------------------------------------------------

    /// Bring the underlying [`FileBuilder`] to its final state: check nothing is
    /// left open, emit the `#subsystem#/MCOS` group if any string objects were
    /// written, attach the `#refs#` group if any refs were created, and stage the
    /// userblock.
    ///
    /// Every exit from the builder goes through here, so the buffered and
    /// streaming ones cannot come to describe different files. `entry_point` names
    /// the caller in the error messages.
    fn finalize(&mut self, entry_point: &str) -> Result<(), MatError> {
        // Compression implies chunked storage, whose chunk indices need the 1.10
        // format. Named here rather than left to the writer's generic refusal,
        // which knows about layouts and not about which MAT option to change.
        if self.options.compression != Compression::None
            && self.options.libver < crate::LibVer::V110
        {
            return Err(MatError::CompressionNeedsNewerFormat);
        }
        if !self.open_structs.is_empty() {
            return Err(MatError::Custom(format!(
                "MatBuilder::{entry_point} called with {} open structs",
                self.open_structs.len()
            )));
        }
        if self.next_target.is_some() {
            return Err(MatError::Custom(format!(
                "MatBuilder::{entry_point} called with a pending cell-element target"
            )));
        }

        if !self.string_object_payload_paths.is_empty() {
            self.emit_subsystem()?;
        }

        if let Some(refs) = self.refs.take() {
            self.file.add_group(refs.finish());
        }

        self.file
            .with_userblock_content(&userblock::header_block(userblock::DEFAULT_DESCRIPTION));
        Ok(())
    }

    /// Replace a write failure with the producer error that actually caused it,
    /// if a [`DataProducer`] stashed one. The writer's provider seam cannot carry
    /// a `MatError`, so without this a producer's own failure would surface as
    /// the placeholder the writer substituted.
    fn attribute_failure(stash: &Arc<Mutex<Option<MatError>>>, fallback: crate::Error) -> MatError {
        match stash.lock() {
            Ok(mut slot) => slot.take().unwrap_or(MatError::Hdf5(fallback)),
            Err(_) => MatError::Hdf5(fallback),
        }
    }

    /// Finalize the file and return its bytes.
    pub fn finish(mut self) -> Result<Vec<u8>, MatError> {
        self.finalize("finish")?;
        let stash = Arc::clone(&self.producer_error);
        self.file
            .finish()
            .map_err(|e| Self::attribute_failure(&stash, e))
    }

    /// Finalize the file and write it to `w` front-to-back, without ever holding
    /// the assembled file in memory.
    ///
    /// Produces byte-for-byte the same file as [`finish`](Self::finish). The sink
    /// is written in ascending-address order and never seeks, so it can be a
    /// socket as readily as a file. A dataset staged through
    /// [`write_blocks`](Self::write_blocks) has its blocks pulled from the
    /// producer here, one at a time.
    ///
    /// A failure partway leaves whatever was already written on the sink. With a
    /// non-seekable sink there is nothing to roll back, so a caller that needs
    /// all-or-nothing should write to a temporary path and rename on success —
    /// which is what [`write`](Self::write) does not do either, for the same
    /// reason.
    ///
    /// # Example
    ///
    /// ```
    /// use hdf5_pure::mat::{MatBuilder, Options};
    ///
    /// let build = || {
    ///     let mut mb = MatBuilder::new(Options::default());
    ///     mb.write_f64("v", &[1, 3], &[1.0, 2.0, 3.0]).unwrap();
    ///     mb
    /// };
    ///
    /// let mut streamed: Vec<u8> = Vec::new();
    /// build().finish_to(&mut streamed).unwrap();
    ///
    /// assert_eq!(build().finish().unwrap(), streamed);
    /// assert_eq!(&streamed[..6], b"MATLAB", "userblock leads the file");
    /// ```
    pub fn finish_to<W: std::io::Write>(mut self, w: W) -> Result<(), MatError> {
        self.finalize("finish_to")?;
        let stash = Arc::clone(&self.producer_error);
        self.file
            .finish_to(w)
            .map_err(|e| Self::attribute_failure(&stash, e))
    }

    /// Finalize the file and stream it to `path`. See
    /// [`finish_to`](Self::finish_to).
    pub fn write<P: AsRef<std::path::Path>>(mut self, path: P) -> Result<(), MatError> {
        self.finalize("write")?;
        let stash = Arc::clone(&self.producer_error);
        self.file
            .write(path)
            .map_err(|e| Self::attribute_failure(&stash, e))
    }

    /// Build the MCOS subsystem. Lifts the layout from the beve writer: 5
    /// helper refs (FileWrapper, canonical empty, unknown-template-A,
    /// alias-int32, unknown-template-B) plus the per-string saveobj payload
    /// paths. Byte-for-byte apart from the [`refs_h5path`] attribute, which
    /// MATLAB writes on all of these but the canonical empty.
    fn emit_subsystem(&mut self) -> Result<(), MatError> {
        // 1. FileWrapper__ metadata blob.
        let metadata = string_object::build_string_filewrapper_metadata(
            self.string_object_payload_paths.len(),
        );
        let metadata_ref = self.alloc_ref_name();
        let metadata_path = format!("{REFS_GROUP}/{metadata_ref}");
        let metadata_shape: [u64; 2] = [1, metadata.len() as u64];
        {
            let refs = self.refs_mut();
            let ds = refs.create_dataset(&metadata_ref);
            ds.set_attr("H5PATH", refs_h5path(&metadata_ref));
            ds.with_u8_data(&metadata).with_shape(&metadata_shape);
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MatClass::UInt8.as_str().into()),
            );
        }

        // 2. Canonical empty.
        let canonical_ref = self.alloc_ref_name();
        let canonical_path = format!("{REFS_GROUP}/{canonical_ref}");
        // The one object under `#refs#` MATLAB leaves without an `H5PATH`.
        self.write_subsystem_empty_marker(
            &canonical_ref,
            &[0, 0],
            "canonical empty",
            None,
            RefsH5Path::Omit,
        );

        // 3. Unknown template A: cell with two empty struct refs.
        let empty_a1 = self.alloc_ref_name();
        let empty_a1_path = format!("{REFS_GROUP}/{empty_a1}");
        self.write_subsystem_empty_marker(&empty_a1, &[1, 0], "struct", None, RefsH5Path::Own);

        let empty_a2 = self.alloc_ref_name();
        let empty_a2_path = format!("{REFS_GROUP}/{empty_a2}");
        self.write_subsystem_empty_marker(&empty_a2, &[1, 0], "struct", None, RefsH5Path::Own);

        let template_a = self.alloc_ref_name();
        let template_a_path = format!("{REFS_GROUP}/{template_a}");
        self.write_subsystem_reference_array(
            &template_a,
            &[2, 1],
            &[empty_a1_path.as_str(), empty_a2_path.as_str()],
            "cell",
        );

        // 4. Alias int32 ref.
        let alias_ref = self.alloc_ref_name();
        let alias_path = format!("{REFS_GROUP}/{alias_ref}");
        {
            let refs = self.refs_mut();
            let ds = refs.create_dataset(&alias_ref);
            ds.set_attr("H5PATH", refs_h5path(&alias_ref));
            ds.with_i32_data(&[0i32, 0]).with_shape(&[1u64, 2]);
            ds.set_attr(
                "MATLAB_class",
                AttrValue::AsciiString(MatClass::Int32.as_str().into()),
            );
        }

        // 5. Unknown template B: cell with two empty struct refs.
        let empty_b1 = self.alloc_ref_name();
        let empty_b1_path = format!("{REFS_GROUP}/{empty_b1}");
        self.write_subsystem_empty_marker(&empty_b1, &[1, 0], "struct", None, RefsH5Path::Own);

        let empty_b2 = self.alloc_ref_name();
        let empty_b2_path = format!("{REFS_GROUP}/{empty_b2}");
        self.write_subsystem_empty_marker(&empty_b2, &[1, 0], "struct", None, RefsH5Path::Own);

        let template_b = self.alloc_ref_name();
        let template_b_path = format!("{REFS_GROUP}/{template_b}");
        self.write_subsystem_reference_array(
            &template_b,
            &[2, 1],
            &[empty_b1_path.as_str(), empty_b2_path.as_str()],
            "cell",
        );

        // 6. Build the MCOS reference array.
        let mut paths: Vec<String> = Vec::with_capacity(5 + self.string_object_payload_paths.len());
        paths.push(metadata_path);
        paths.push(canonical_path);
        paths.extend(self.string_object_payload_paths.iter().cloned());
        paths.push(template_a_path);
        paths.push(alias_path);
        paths.push(template_b_path);

        let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        let mut subsystem_group = self.file.create_group(SUBSYSTEM_GROUP);
        let ds = subsystem_group.create_dataset("MCOS");
        ds.with_path_references(&path_refs)
            .with_shape(&[1u64, paths.len() as u64]);
        ds.set_attr(
            "MATLAB_class",
            AttrValue::AsciiString(string_object::MATLAB_CLASS_FILEWRAPPER.into()),
        );
        ds.set_attr(
            "MATLAB_object_decode",
            AttrValue::I32(string_object::MATLAB_OBJECT_DECODE_OPAQUE),
        );
        self.file.add_group(subsystem_group.finish());
        Ok(())
    }

    /// Subsystem helper refs always use the data-as-dims encoding regardless
    /// of `Options::empty_marker_encoding`: real MATLAB writes them this way
    /// in the `mxOPAQUE_CLASS` machinery, and we lift the layout byte-for-byte.
    fn write_subsystem_empty_marker(
        &mut self,
        ref_name: &str,
        matlab_dims: &[usize],
        class: &str,
        int_decode: Option<i32>,
        h5path: RefsH5Path,
    ) {
        let mut dim_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let n = matlab_dims.len();
        let refs = self.refs_mut();
        let ds = refs.create_dataset(ref_name);
        if h5path == RefsH5Path::Own {
            ds.set_attr("H5PATH", refs_h5path(ref_name));
        }
        if n > STORAGE_DIMS_BUF_LEN {
            let dim_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();
            ds.with_u64_data(&dim_data).with_shape(&[n as u64]);
        } else {
            for (slot, &d) in dim_buf[..n].iter_mut().zip(matlab_dims) {
                *slot = d as u64;
            }
            ds.with_u64_data(&dim_buf[..n]).with_shape(&[n as u64]);
        }
        ds.set_attr("MATLAB_class", AttrValue::AsciiString(class.into()));
        ds.set_attr("MATLAB_empty", AttrValue::U32(1));
        if let Some(d) = int_decode {
            ds.set_attr("MATLAB_int_decode", AttrValue::I32(d));
        }
    }

    fn write_subsystem_reference_array(
        &mut self,
        ref_name: &str,
        matlab_dims: &[usize],
        paths: &[&str],
        class: &str,
    ) {
        let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
        let storage = storage_dims_u64_into(matlab_dims, &mut storage_buf);
        let refs = self.refs_mut();
        let ds = refs.create_dataset(ref_name);
        ds.set_attr("H5PATH", refs_h5path(ref_name));
        ds.with_path_references(paths).with_shape(storage);
        ds.set_attr("MATLAB_class", AttrValue::AsciiString(class.into()));
    }
}

/// Where to put the next dataset/group, as resolved from current scope and
/// `next_target`.
enum TargetForName {
    Root(String),
    Struct { name: String, index: usize },
    Ref(String),
}

/// Apply the configured deflate compression to `ds`, if any.
fn apply_deflate(ds: &mut DatasetBuilder, compression: Compression) {
    if let Compression::Deflate { level, shuffle } = compression {
        if shuffle {
            ds.with_shuffle();
        }
        ds.with_deflate(level as u32);
    }
}

/// Write the storage an empty value takes under `encoding` — the data and the
/// shape only. The caller sets `MATLAB_class`, `MATLAB_empty`, and any decode
/// flag, which do not vary with the encoding.
///
/// This is the one place the encoding is acted on, and both MAT serde emitters
/// route through it. They are required to produce identical bytes for the same
/// value under default options (`tests/serde_roundtrip.rs`), and this is exactly
/// where they last came apart: the no-options emitter had `ZeroElement` written
/// into it as a constant, so moving the default to `DataAsDims` changed one
/// emitter and not the other.
pub(crate) fn emit_empty_storage(
    ds: &mut DatasetBuilder,
    encoding: EmptyMarkerEncoding,
    class: MatClass,
    matlab_dims: &[usize],
) {
    match encoding {
        EmptyMarkerEncoding::ZeroElement => {
            let mut storage_buf = [0u64; STORAGE_DIMS_BUF_LEN];
            let shape = storage_dims_u64_into(matlab_dims, &mut storage_buf);
            emit_zero_element(ds, class, shape);
        }
        EmptyMarkerEncoding::DataAsDims => {
            if matlab_dims.len() > STORAGE_DIMS_BUF_LEN {
                let dim_data: Vec<u64> = matlab_dims.iter().map(|&d| d as u64).collect();
                ds.with_u64_data(&dim_data)
                    .with_shape(&[dim_data.len() as u64]);
            } else {
                let mut dim_buf = [0u64; STORAGE_DIMS_BUF_LEN];
                let n = matlab_dims.len();
                for (slot, &d) in dim_buf[..n].iter_mut().zip(matlab_dims) {
                    *slot = d as u64;
                }
                ds.with_u64_data(&dim_buf[..n]).with_shape(&[n as u64]);
            }
        }
    }
}

/// Write the zero-element dataset that carries an empty marker under
/// [`EmptyMarkerEncoding::ZeroElement`], with the element type the class
/// implies. Reached through [`emit_empty_storage`], which picks the encoding.
pub(crate) fn emit_zero_element(ds: &mut DatasetBuilder, class: MatClass, shape: &[u64]) {
    match class {
        MatClass::Double => {
            ds.with_f64_data(&[]).with_shape(shape);
        }
        MatClass::Single => {
            ds.with_f32_data(&[]).with_shape(shape);
        }
        MatClass::Int8 => {
            ds.with_i8_data(&[]).with_shape(shape);
        }
        MatClass::Int16 => {
            ds.with_i16_data(&[]).with_shape(shape);
        }
        MatClass::Int32 => {
            ds.with_i32_data(&[]).with_shape(shape);
        }
        MatClass::Int64 => {
            ds.with_i64_data(&[]).with_shape(shape);
        }
        MatClass::UInt8 | MatClass::Logical => {
            ds.with_u8_data(&[]).with_shape(shape);
        }
        MatClass::UInt16 | MatClass::Char => {
            ds.with_u16_data(&[]).with_shape(shape);
        }
        MatClass::UInt32 => {
            ds.with_u32_data(&[]).with_shape(shape);
        }
        // A container class has no element type to inherit, so the width is a
        // free choice. `uint64` is what the reference library writes for an
        // empty marker under the other encoding, where the payload is the dims
        // vector, which keeps one width across both encodings.
        MatClass::UInt64 | MatClass::Cell | MatClass::Struct => {
            ds.with_u64_data(&[]).with_shape(shape);
        }
    }
}

/// Scoped writer inside a struct group. Mirrors [`MatBuilder`]'s public write
/// methods but writes to the parent struct.
pub struct StructWriter<'a> {
    mb: &'a mut MatBuilder,
}

impl<'a> StructWriter<'a> {
    /// Borrow the underlying [`MatBuilder`]. Use this when a generic walker
    /// needs to write into the current scope without going through the
    /// struct-specific forwarding methods. The builder's internal scope
    /// stack ensures writes target the open struct group.
    #[inline]
    pub fn builder(&mut self) -> &mut MatBuilder {
        self.mb
    }

    #[inline]
    pub fn write_scalar_logical(&mut self, name: &str, value: bool) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_logical(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_f64(&mut self, name: &str, value: f64) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_f64(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_f32(&mut self, name: &str, value: f32) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_f32(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_i8(&mut self, name: &str, value: i8) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_i8(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_i16(&mut self, name: &str, value: i16) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_i16(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_i32(&mut self, name: &str, value: i32) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_i32(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_i64(&mut self, name: &str, value: i64) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_i64(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_u8(&mut self, name: &str, value: u8) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_u8(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_u16(&mut self, name: &str, value: u16) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_u16(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_u32(&mut self, name: &str, value: u32) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_u32(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_scalar_u64(&mut self, name: &str, value: u64) -> Result<&mut Self, MatError> {
        self.mb.write_scalar_u64(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_f64(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[f64],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_f64(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_f32(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[f32],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_f32(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_i8(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[i8],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_i8(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_i16(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[i16],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_i16(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_i32(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[i32],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_i32(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_i64(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[i64],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_i64(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_u8(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u8],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_u8(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_u16(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u16],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_u16(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_u32(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u32],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_u32(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_u64(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u64],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_u64(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_logical(
        &mut self,
        name: &str,
        dims: &[usize],
        data: &[u8],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_logical(name, dims, data)?;
        Ok(self)
    }
    #[inline]
    pub fn write_char(&mut self, name: &str, value: &str) -> Result<&mut Self, MatError> {
        self.mb.write_char(name, value)?;
        Ok(self)
    }
    #[inline]
    pub fn write_string_object(
        &mut self,
        name: &str,
        values: &[String],
        dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_string_object(name, values, dims)?;
        Ok(self)
    }
    #[inline]
    pub fn write_empty(
        &mut self,
        name: &str,
        class: MatClass,
        dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        self.mb.write_empty(name, class, dims)?;
        Ok(self)
    }
    #[inline]
    pub fn write_empty_struct_array(&mut self, name: &str) -> Result<&mut Self, MatError> {
        self.mb.write_empty_struct_array(name)?;
        Ok(self)
    }
    pub fn struct_<F>(&mut self, name: &str, fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut StructWriter) -> Result<(), MatError>,
    {
        self.mb.struct_(name, fill)?;
        Ok(self)
    }
    pub fn cell<F>(&mut self, name: &str, dims: &[usize], fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut CellWriter) -> Result<(), MatError>,
    {
        self.mb.cell(name, dims, fill)?;
        Ok(self)
    }

    /// MATLAB shape for a 1-D vector.
    #[inline]
    pub fn vector_dims(&self, len: usize) -> [usize; 2] {
        self.mb.vector_dims(len)
    }
    /// MATLAB shape for multi-dimensional extents.
    #[inline]
    pub fn matrix_dims(&self, extents: &[usize]) -> Vec<usize> {
        self.mb.matrix_dims(extents)
    }
    /// Access the underlying options.
    #[inline]
    pub fn options(&self) -> &Options {
        self.mb.options()
    }
    /// Convenience: configured [`StringClass`](crate::mat::options::StringClass).
    #[inline]
    pub fn string_class(&self) -> crate::mat::options::StringClass {
        self.options().string_class
    }
}

/// Scoped writer inside a cell array. Each `push_*` allocates a fresh
/// `#refs#/ref_NNNN` and records its absolute path. The cell's parent dataset
/// is emitted when the closure returns.
pub struct CellWriter<'a> {
    mb: &'a mut MatBuilder,
    paths: &'a mut Vec<String>,
}

impl<'a> CellWriter<'a> {
    fn arm(&mut self) -> String {
        let ref_name = self.mb.alloc_ref_name();
        self.mb.next_target = Some(NextTarget {
            ref_name: ref_name.clone(),
        });
        ref_name
    }

    fn record(&mut self, ref_name: String) {
        // Defensive: if the write didn't consume next_target (shouldn't
        // happen), drop it so the next push starts clean.
        let _ = self.mb.next_target.take();
        self.paths.push(format!("{REFS_GROUP}/{ref_name}"));
    }

    pub fn push_scalar_logical(&mut self, value: bool) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_logical(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_f64(&mut self, value: f64) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_f64(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_f32(&mut self, value: f32) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_f32(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_u8(&mut self, value: u8) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_u8(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_u16(&mut self, value: u16) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_u16(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_u32(&mut self, value: u32) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_u32(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_u64(&mut self, value: u64) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_u64(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_i8(&mut self, value: i8) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_i8(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_i16(&mut self, value: i16) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_i16(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_i32(&mut self, value: i32) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_i32(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_scalar_i64(&mut self, value: i64) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_scalar_i64(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_string(&mut self, value: &str) -> Result<&mut Self, MatError> {
        // MATLAB string scalar: dims [1, 1].
        let r = self.arm();
        self.mb
            .write_string_object(&r, &[value.to_owned()], &[1, 1])?;
        self.record(r);
        Ok(self)
    }
    pub fn push_char(&mut self, value: &str) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_char(&r, value)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_f64(&mut self, dims: &[usize], data: &[f64]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_f64(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_f32(&mut self, dims: &[usize], data: &[f32]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_f32(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_i8(&mut self, dims: &[usize], data: &[i8]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_i8(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_i16(&mut self, dims: &[usize], data: &[i16]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_i16(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_i32(&mut self, dims: &[usize], data: &[i32]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_i32(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_i64(&mut self, dims: &[usize], data: &[i64]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_i64(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_u8(&mut self, dims: &[usize], data: &[u8]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_u8(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_u16(&mut self, dims: &[usize], data: &[u16]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_u16(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_u32(&mut self, dims: &[usize], data: &[u32]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_u32(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_u64(&mut self, dims: &[usize], data: &[u64]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_u64(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_logical(&mut self, dims: &[usize], data: &[u8]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_logical(&r, dims, data)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_string_object(
        &mut self,
        values: &[String],
        dims: &[usize],
    ) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_string_object(&r, values, dims)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_empty(&mut self, class: MatClass, dims: &[usize]) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_empty(&r, class, dims)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_empty_struct_array(&mut self) -> Result<&mut Self, MatError> {
        let r = self.arm();
        self.mb.write_empty_struct_array(&r)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_struct<F>(&mut self, fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut StructWriter) -> Result<(), MatError>,
    {
        let r = self.arm();
        self.mb.struct_(&r, fill)?;
        self.record(r);
        Ok(self)
    }
    pub fn push_cell<F>(&mut self, dims: &[usize], fill: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut CellWriter) -> Result<(), MatError>,
    {
        let r = self.arm();
        self.mb.cell(&r, dims, fill)?;
        self.record(r);
        Ok(self)
    }

    /// Push an explicit reference to an existing path (e.g. a previously
    /// allocated `#refs#/ref_NNNN` written via [`MatBuilder::alloc_ref_name`]
    /// and out-of-band logic). For advanced use only.
    pub fn push_path(&mut self, path: String) -> &mut Self {
        self.paths.push(path);
        self
    }

    /// Allocate a fresh ref and run `build` with the underlying `MatBuilder`,
    /// armed so the next `write_*` / `struct_` / `cell` call inside `build`
    /// targets `#refs#/ref_NNNN`. The build closure is responsible for
    /// performing exactly one write/struct/cell operation.
    pub fn push_with<F>(&mut self, build: F) -> Result<&mut Self, MatError>
    where
        F: FnOnce(&mut MatBuilder) -> Result<(), MatError>,
    {
        let r = self.arm();
        let res = build(self.mb);
        // `record` clears any leftover next_target as a safety net.
        self.record(r);
        res?;
        Ok(self)
    }

    /// Access the configured options.
    #[inline]
    pub fn options(&self) -> &Options {
        self.mb.options()
    }

    /// MATLAB shape for a 1-D vector under the configured mode.
    #[inline]
    pub fn vector_dims(&self, len: usize) -> [usize; 2] {
        self.mb.vector_dims(len)
    }

    /// MATLAB shape for multi-dimensional extents under the configured mode.
    #[inline]
    pub fn matrix_dims(&self, extents: &[usize]) -> Vec<usize> {
        self.mb.matrix_dims(extents)
    }
}

/// The complex writers, one per component class, on all three write scopes.
///
/// Every one of them is [`MatBuilder::write_complex_of`] with the class filled
/// in, so a new component width is a line in the invocation below rather than
/// three hand-written methods that can drift apart.
macro_rules! complex_writers {
    ($($write:ident / $push:ident => $ty:ty, $class:ident, $matlab:literal),* $(,)?) => {
        impl MatBuilder {
            $(
                #[doc = concat!(
                    "Write a complex `", stringify!($ty), "` array (MATLAB class `",
                    $matlab, "`)."
                )]
                ///
                /// Empty inputs produce a zero-element compound dataset with no
                /// `MATLAB_empty` marker, which MATLAB reads back as an empty
                /// complex array of this class. Non-empty inputs honor the
                /// configured compression settings.
                pub fn $write(
                    &mut self,
                    name: &str,
                    matlab_dims: &[usize],
                    data: &[($ty, $ty)],
                ) -> Result<&mut Self, MatError> {
                    self.write_complex_of(name, MatClass::$class, matlab_dims, data)
                }
            )*
        }

        impl StructWriter<'_> {
            $(
                #[doc = concat!(
                    "Write a complex `", stringify!($ty),
                    "` array into the open struct. See [`MatBuilder::",
                    stringify!($write), "`]."
                )]
                #[inline]
                pub fn $write(
                    &mut self,
                    name: &str,
                    dims: &[usize],
                    data: &[($ty, $ty)],
                ) -> Result<&mut Self, MatError> {
                    self.mb.$write(name, dims, data)?;
                    Ok(self)
                }
            )*
        }

        impl CellWriter<'_> {
            $(
                #[doc = concat!(
                    "Push a complex `", stringify!($ty),
                    "` array as the next cell element. See [`MatBuilder::",
                    stringify!($write), "`]."
                )]
                pub fn $push(
                    &mut self,
                    dims: &[usize],
                    data: &[($ty, $ty)],
                ) -> Result<&mut Self, MatError> {
                    let r = self.arm();
                    self.mb.$write(&r, dims, data)?;
                    self.record(r);
                    Ok(self)
                }
            )*
        }
    };
}

complex_writers! {
    write_complex_f64 / push_complex_f64 => f64, Double, "double",
    write_complex_f32 / push_complex_f32 => f32, Single, "single",
    write_complex_i64 / push_complex_i64 => i64, Int64, "int64",
    write_complex_i32 / push_complex_i32 => i32, Int32, "int32",
    write_complex_i16 / push_complex_i16 => i16, Int16, "int16",
    write_complex_i8 / push_complex_i8 => i8, Int8, "int8",
    write_complex_u64 / push_complex_u64 => u64, UInt64, "uint64",
    write_complex_u32 / push_complex_u32 => u32, UInt32, "uint32",
    write_complex_u16 / push_complex_u16 => u16, UInt16, "uint16",
    write_complex_u8 / push_complex_u8 => u8, UInt8, "uint8",
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::File;
    use crate::types::AttrValue as ReaderAttr;

    /// A `.mat` fixture in a directory of its own. The directory goes away with
    /// the returned value, so bind it for as long as the file is needed.
    ///
    /// Replaces a nanosecond-stamped name in the shared temporary directory,
    /// which no two runs collided on but no run ever cleaned up (issue #334).
    fn temp_path(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(format!("hdf5pure-mat-{name}.mat"));
        (dir, path)
    }

    fn read_class(file: &File, ds_path: &str) -> String {
        let ds = file.dataset(ds_path).unwrap();
        let attrs = ds.attrs().unwrap();
        match &attrs["MATLAB_class"] {
            ReaderAttr::AsciiString(s) | ReaderAttr::String(s) => s.clone(),
            other => panic!("unexpected class: {other:?}"),
        }
    }

    #[test]
    fn scalar_f64_at_root() {
        let mut mb = MatBuilder::new(Options::default());
        mb.write_scalar_f64("x", 1.5).unwrap();
        let bytes = mb.finish().unwrap();

        let (_dir, path) = temp_path("scalar-f64");
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let ds = file.dataset("x").unwrap();
        assert_eq!(ds.read_f64().unwrap(), vec![1.5]);
        assert_eq!(read_class(&file, "x"), "double");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn nested_struct_writes_fields() {
        let mut mb = MatBuilder::new(Options::default());
        mb.struct_("payload", |s| {
            s.write_scalar_u32("answer", 7)?;
            s.write_char("label", "hello")?;
            Ok(())
        })
        .unwrap();
        let bytes = mb.finish().unwrap();

        let (_dir, path) = temp_path("struct");
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let group = file.group("payload").unwrap();
        let attrs = group.attrs().unwrap();
        let class = match &attrs["MATLAB_class"] {
            ReaderAttr::AsciiString(s) | ReaderAttr::String(s) => s.clone(),
            other => panic!("unexpected: {other:?}"),
        };
        assert_eq!(class, "struct");
        assert_eq!(read_class(&file, "payload/answer"), "uint32");
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cell_with_two_refs() {
        let mut mb = MatBuilder::new(Options::default());
        mb.cell("c", &[2, 1], |cw| {
            cw.push_scalar_u8(1)?;
            cw.push_scalar_u8(2)?;
            Ok(())
        })
        .unwrap();
        let bytes = mb.finish().unwrap();

        let (_dir, path) = temp_path("cell");
        std::fs::write(&path, &bytes).unwrap();
        let file = File::open(&path).unwrap();
        let cls = read_class(&file, "c");
        assert_eq!(cls, "cell");
        assert_eq!(read_class(&file, "#refs#/ref_0000000000000000"), "uint8");
        assert_eq!(read_class(&file, "#refs#/ref_0000000000000001"), "uint8");
        std::fs::remove_file(path).unwrap();
    }

    #[derive(Debug, PartialEq)]
    struct EmbedderError(u32);

    impl std::fmt::Display for EmbedderError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "embedder failed with code {}", self.0)
        }
    }

    impl std::error::Error for EmbedderError {}

    /// A caller's own error type survives two nested closures, each of which
    /// runs its own cleanup on the way out: `cell` resolves its parent target
    /// before `fill`, and `push_with` records the reference it armed before it
    /// propagates. The doctest on [`MatError::Source`] covers `struct_`, the
    /// third shape.
    #[test]
    fn a_caller_error_crosses_the_closure_boundary_whole() {
        use std::error::Error;

        let mut mb = MatBuilder::new(Options::default());
        let err = mb
            .cell("c", &[1, 1], |cw| {
                cw.push_with(|_| Err(MatError::from_source(EmbedderError(7))))?;
                Ok(())
            })
            .err()
            .expect("the closure failed");

        assert_eq!(err.to_string(), "embedder failed with code 7");
        assert_eq!(
            err.source().unwrap().downcast_ref::<EmbedderError>(),
            Some(&EmbedderError(7))
        );
    }
}
