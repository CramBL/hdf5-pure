[`File::open_rw`](crate::File::open_rw) opens an existing HDF5 file for reading **and** writing, and edits it through owned [`Dataset`](crate::Dataset) and [`Group`](crate::Group) handles that reach every object by name: adding, copying, or deleting objects, or editing attributes, without rewriting the file from scratch. New data and rebuilt object headers are allocated according to the file-space strategy, and the superblock is repointed last, so the cost is proportional to what changes and independent of the file size, and a failed commit leaves the original file valid.

This page mirrors [`examples/edit_in_place.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/edit_in_place.rs). Run it with:

```console
$ cargo run --example edit_in_place
```

## Choosing a write path

[`File::open_rw`](crate::File::open_rw) selects the backing from the file. Eligible latest-format files use bounded memory: the configured caches and the content an edit builds. Files with a pre-v2 superblock, a userblock, or non-persistent [`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page) allocation require a whole-file mirror, using memory proportional to file size. File size alone does not affect the selection. Both backings allocate according to the file-space strategy and publish rebuilt trees through the superblock.

[`File::edit_backing()`](crate::File::edit_backing) reports the backing a file got, and [`FileAccessProperties::with_memory_strategy`](crate::FileAccessProperties::with_memory_strategy) demands one:

| [`MemoryStrategy`](crate::MemoryStrategy) | Effect |
| --- | --- |
| unset (the default) | Prefer bounded, and fall back to the mirror for a file the bounded engine cannot edit |
| [`Bounded`](crate::MemoryStrategy::Bounded) | Never build a mirror, and reject such a file with [`Error::EditUnsupported`](crate::Error::EditUnsupported) |
| [`Mirrored`](crate::MemoryStrategy::Mirrored) | Always build the mirror, whatever the file looks like |

[`File::edit_backing()`](crate::File::edit_backing) reports an [`EditBacking`](crate::EditBacking), [`Bounded`](crate::EditBacking::Bounded) or [`Mirrored`](crate::EditBacking::Mirrored), where the caller supplied a [`MemoryStrategy`](crate::MemoryStrategy): [`Auto`](crate::MemoryStrategy::Auto) is a preference between the two backings and never an outcome. `.into()` converts an [`EditBacking`](crate::EditBacking) back into the [`MemoryStrategy`](crate::MemoryStrategy) that pins a later reopen to it.

[`File::open_swmr_writer`](crate::File::open_swmr_writer) always mirrors, so it accepts [`Auto`](crate::MemoryStrategy::Auto) and [`Mirrored`](crate::MemoryStrategy::Mirrored), both of which the mirror satisfies, and rejects an explicit [`Bounded`](crate::MemoryStrategy::Bounded) at the open.

The two backings are the same engine and the same edit vocabulary, reads, immediate [`Dataset::append`](crate::Dataset::append) / [`append_raw`](crate::Dataset::append_raw), buffered [`Dataset::buffered_appender`](crate::Dataset::buffered_appender), [`Dataset::write`](crate::Dataset::write), [`append_staged`](crate::Dataset::append_staged), [`set_attr`](crate::Dataset::set_attr) / [`remove_attr`](crate::Dataset::remove_attr), [`Group::create_dataset`](crate::Group::create_dataset) / [`create_group`](crate::Group::create_group) / [`create_group_with`](crate::Group::create_group_with) / [`delete`](crate::Group::delete), [`File::copy`](crate::File::copy) / [`copy_from`](crate::File::copy_from), [`commit`](crate::File::commit), [`reusable_space_stats`](crate::File::reusable_space_stats), and [`reusable_free_sections`](crate::File::reusable_free_sections), with one trade between them. A large [`Dataset::append`](crate::Dataset::append) is one crash-atomic apply on the mirror and several ~1 MiB whole-chunk batches when bounded, so a crash mid-call there leaves a valid shorter dataset. A commit's resident memory follows the backing too: the edit bounds it, and the file's size does not, with [`File::copy`](crate::File::copy) the exception, since copying an object reads the whole of it into memory first.

One caveat inside either backing: the immediate [`Dataset::append`](crate::Dataset::append) is stricter than the staged surface. It needs a latest-format (v2/v3-superblock) file with no userblock, and the error it returns ([`Error::AppendInPlaceUnsupported`](crate::Error::AppendInPlaceUnsupported)) points at [`Dataset::append_staged`](crate::Dataset::append_staged) as the fallback.

The file's file-space strategy gates what an edit can do at all, whichever backing holds it:

| File-space strategy | Staged edits + [`commit`](crate::File::commit) | Immediate append |
| --- | --- | --- |
| None recorded, or [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) with `persist = false` | Yes | Yes |
| [`Aggr`](crate::FileSpaceStrategy::Aggr) / [`None`](crate::FileSpaceStrategy::None), with persistence ignored | Yes | Yes |
| [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr) with `persist = true` | Yes: freed space is recorded on disk | Yes: managers rewritten at [`close`](crate::File::close) |
| [`Page`](crate::FileSpaceStrategy::Page) with `persist = true` | Yes: page-aware commit, per-page-type managers rewritten | Yes: appends stay page-homogeneous |
| [`Page`](crate::FileSpaceStrategy::Page) with `persist = false` | Mirrored only, in fresh typed pages, with the restrictions below | Rejected |

With PAGE persistence disabled, [`File::open_rw`](crate::File::open_rw) selects the mirror. An explicit [`MemoryStrategy::Bounded`](crate::MemoryStrategy::Bounded) is rejected with [`Error::EditUnsupported`](crate::Error::EditUnsupported). Before writing or consuming staged edits, a commit requires supported attribute storage and a complete tree proved free of object references. Dense attributes are rejected. Immediate append, SWMR writing, and page buffering are unsupported.

The editor appends allocations in typed pages at or beyond the first page boundary covering both the original EOA and physical extent, computed in stored addresses. All existing pages, holes, and tails remain unavailable for allocation. Deletion and replacement retain old bytes, including across repeated commits and reopen. The editor preserves the File Space Info message with `persist = false`, keeps the manager address list empty, and leaves the userblock unchanged. It writes new content before publishing the root and EOA through the superblock, using the selected synchronization policy. This mode can grow the file on each commit.

An I/O error can occur after the complete batch is published. While PAGE publication recovery is unresolved, session reads are insufficient to determine the publication outcome. A subsequent commit first drains retained writes under the selected synchronization policy and verifies the issued superblock against the accepted image and expected framing. Explicit [`File::sync`](crate::File::sync) performs that verification after a durable synchronization barrier. Successful recovery adopts the verified root. Recovery failures leave later staged edits pending. Byte or framing mismatches return [`Error::EditUnsupported`](crate::Error::EditUnsupported) and require closing and reopening the file. An interrupted file can fail to reopen.

A paged commit keeps each page homogeneous, so raw data and file metadata never share a page, apart from a chunked dataset's index, which travels with its chunk data. It also page-aligns the end of allocation, so libhdf5 reopens the result as a paged file. With persistence enabled, a free hole belongs to the page type of the page it sits in, and an allocation draws only on holes of its own type. A page holding nothing at all belongs to neither, so either type may open one. The page types are set out in the [file-space strategy](crate::_guide::file_space) guide.

For a brand-new file, use [`FileBuilder`](crate::FileBuilder), which [Writing files](crate::_guide::writing) walks. To append while readers are live, use the [SWMR writer](crate::_guide::swmr). To compact a file or drop objects across a reopen, use [`repack`](crate::repack()). The [file-creation](crate::FileCreateProperties#supported-properties) and [file-access](crate::FileAccessProperties#supported-properties) property tables list what each path supports.

## Staging and committing edits

An edit session is transactional: you stage operations on an open file, then apply them all at once with [`commit()`](crate::File::commit). The editor records staged operations in memory.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
# let mut sensors = builder.create_group("sensors");
# sensors.create_dataset("pressure").with_f32_data(&[101.3]);
# builder.add_group(sensors.finish());
# builder.write(&path)?;
use hdf5_pure::{AttrValue, File};

let file = File::open_rw(&path)?;
let root = file.root();

root.create_group_with("run2", |g| {
    g.set_attr("kind", AttrValue::AsciiString("trial".into()));
})?;
root.create_dataset("run2/signal", |b| {
    b.with_f64_data(&[1.0, 2.0, 3.0]);
})?;
file.copy("temperature", "temperature_backup")?; // H5Ocopy
root.delete("sensors/pressure")?;                // H5Ldelete

file.commit()?; // apply everything in place
# drop(root);
# drop(file);
# let reopened = File::open(&path)?;
# assert_eq!(reopened.dataset("run2/signal")?.read_f64()?, vec![1.0, 2.0, 3.0]);
# Ok::<(), hdf5_pure::Error>(())
```

After a successful [`commit()`](crate::File::commit), the staged set is cleared and the open file can be reused for further edits.

### Handles across a commit

[`Dataset`](crate::Dataset) and [`Group`](crate::Group) handles stay usable across [`commit()`](crate::File::commit). A commit relocates object headers, and each handle looks its object up again by path on its next use, so a handle held in a struct or a cache reads the file the commit left, never the copy it moved away from. An edit made through *another* handle to the same object reaches it the same way. Both types are `Clone`.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# let mut run2 = builder.create_group("run2");
# run2.create_dataset("signal").with_f64_data(&[1.0, 2.0, 3.0]);
# builder.add_group(run2.finish());
# builder.write(&path)?;
use hdf5_pure::{AttrValue, File};

let file = File::open_rw(&path)?;
let mut signal = file.dataset("run2/signal")?;

signal.set_attr("units", AttrValue::AsciiString("V".into()))?;
file.commit()?;

assert_eq!(signal.attrs()?["units"], AttrValue::AsciiString("V".into())); // same handle, new file
# Ok::<(), hdf5_pure::Error>(())
```

Three cases return an error:

- **Reading** through a handle onto an object the commit **deleted** fails the way opening it by name would ([`FormatError::PathNotFound`](crate::FormatError::PathNotFound)). Onto one the commit **replaced with an object of a different kind**, it is [`Error::NotADataset`](crate::Error::NotADataset) or [`Error::NotAGroup`](crate::Error::NotAGroup). The bytes a deleted object leaves behind still parse as the object that left them, so reading them would return data the file no longer holds. Its write methods address the file by path, so they still stage, and the commit rejects them.
- A handle reached by **object reference** ([`Dataset::dereference`](crate::Dataset::dereference)) stores an object-header address. It returns [`Error::StaleHandle`](crate::Error::StaleHandle) after a commit, staging an edit, closing the file, or publication recovery that changes the root. Dereference again from a fresh read after these operations. Ordinary [`sync`](crate::File::sync) and immediate [`append`](crate::Dataset::append) preserve the handle when the root stays unchanged.
- A handle onto an object the session has **staged** and not yet committed returns [`Error::NotCommitted`](crate::Error::NotCommitted) for anything that reads bytes, and serves everything else. The commit is what turns it into an ordinary handle. See [Staged objects are addressable straight away](#staged-objects-are-addressable-straight-away).

## Operations

| Method | Effect | HDF5 analog |
| --- | --- | --- |
| [`File::open_rw(path)`](crate::File::open_rw) | Open an existing file for reading and writing | none |
| [`Group::create_group(name)`](crate::Group::create_group) | Stage a new empty group, returning a [`Group`](crate::Group) handle onto it | `H5Gcreate` |
| [`Group::create_group_with(name, build)`](crate::Group::create_group_with) | Stage a new group, configured through `build` (attributes, nested objects), returning a [`Group`](crate::Group) handle onto it | `H5Gcreate` |
| [`Group::create_dataset(name, build)`](crate::Group::create_dataset) | Stage a new dataset, configured through a [`DatasetBuilder`](crate::DatasetBuilder), returning a [`Dataset`](crate::Dataset) handle onto it | `H5Dcreate` |
| [`Dataset::append_staged(build)`](crate::Dataset::append_staged) | Stage appending elements along axis 0 of an existing chunked, unlimited dataset, via an [`AppendBuilder`](crate::AppendBuilder) | `H5Dset_extent` + write |
| [`Dataset::append(data)`](crate::Dataset::append) | Append immediately and durably, no [`commit`](crate::File::commit) needed | `H5Dset_extent` + write |
| [`Dataset::buffered_appender()`](crate::Dataset::buffered_appender) | Buffer appended elements and write them a whole chunk at a time | `H5Pset_chunk_cache` (write side) |
| [`Dataset::write(data)`](crate::Dataset::write) / [`write_staged(build)`](crate::Dataset::write_staged) | Stage a value overwrite of the same datatype and shape | `H5Dwrite` |
| [`Group::set_attr(name, value)`](crate::Group::set_attr) / [`Dataset::set_attr`](crate::Dataset::set_attr) | Stage adding or replacing a compact attribute | none |
| [`Group::remove_attr(name)`](crate::Group::remove_attr) / [`Dataset::remove_attr`](crate::Dataset::remove_attr) | Stage removing a compact attribute | none |
| [`File::copy(src, dst)`](crate::File::copy) | Stage a deep copy of a dataset or whole group subtree within this file | `H5Ocopy` |
| [`File::copy_from(source, src, dst)`](crate::File::copy_from) | Copy a dataset or subtree out of another open [`File`](crate::File) into this one | `H5Ocopy` (across files) |
| [`Group::delete(name)`](crate::Group::delete) | Stage removing the link at `name` (and, for a group, its whole subtree) | `H5Ldelete` |
| [`File::commit()`](crate::File::commit) | Apply all staged operations in place and flush | none |

[`create_dataset`](crate::Group::create_dataset) hands you the same [`DatasetBuilder`](crate::DatasetBuilder) used by [`FileBuilder`](crate::FileBuilder), so you configure the new dataset exactly as you would when creating a file from scratch:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# hdf5_pure::FileBuilder::new().write(&path)?;
# let file = hdf5_pure::File::open_rw(&path)?;
# let root = file.root();
# root.create_group("run2")?;
root.create_dataset("run2/signal", |b| {
    b.with_f64_data(&[1.0, 2.0, 3.0]);
})?;
# file.commit()?;
# Ok::<(), hdf5_pure::Error>(())
```

A new dataset may be created **empty**, with zero elements, chunked, with or without an unlimited maximum, which is how a schema-first writer declares its columns before any data has arrived and then grows each one with [`append_staged`](#appending-to-an-unlimited-dataset) as batches come in:

```rust
# use hdf5_pure::MaxExtent;
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("schema.h5");
# hdf5_pure::FileBuilder::new().write(&path)?;
# let file = hdf5_pure::File::open_rw(&path)?;
# let root = file.root();
root.create_dataset("col", |b| {
    b.with_f64_data(&[])
        .with_shape(&[0])
        .with_maxshape(&[MaxExtent::Unlimited])
        .with_chunks(&[512]);
})?;
# file.commit()?;
# Ok::<(), hdf5_pure::Error>(())
```

The chunk dimensions have to be given: auto-chunking derives them from the shape, and a zero-element shape has none to derive from.

The [`create_group_with`](crate::Group::create_group_with) / [`create_dataset`](crate::Group::create_dataset) / [`write_staged`](crate::Dataset::write_staged) / [`append_staged`](crate::Dataset::append_staged) closures configure a builder and leave the file itself alone, so a closure may read the same [`File`](crate::File): staging a dataset whose contents depend on one already there works. What it reads is the file as it was before the call, since the closure runs before its edits are recorded.

[`set_attr`](crate::Group::set_attr) takes an [`AttrValue`](crate::AttrValue), fixed-size or variable-length. [`File::root()`](crate::File::root) returns the root group. Attributes go in the rebuilt header while they fit it compactly. Past that, at more than eight of them, or at one too large for an object-header message, the whole set moves to a fractal heap, and an object already storing its attributes in one is rebuilt the same way ([#102](https://github.com/CramBL/hdf5-pure/issues/102)). An object stays dense once it is dense, and the heap goes away only when its last attribute is removed. A rebuilt heap re-encodes the attributes it carries over: name, datatype, dataspace and value survive (the name's declared character set is rewritten as ASCII), and the edit reclaims the heap it replaces. A heap that has its own free-space manager stays as dead bytes for [`repack`](crate::repack()) to recover. An object that tracks attribute creation order keeps it across the rebuild, creation-order index and all ([#416](https://github.com/CramBL/hdf5-pure/issues/416)). A dataset or group *created* in place may carry a dense set the same way.

An object that tracks **attribute creation order** is edited like any other ([#416](https://github.com/CramBL/hdf5-pure/issues/416)). A new attribute takes the object's next creation index, overwriting one keeps the index it had, and deleting one leaves a gap and renumbers nothing, so an iteration by creation order reads the same afterwards. A group that tracks **link** creation order works the same way: a dataset or group created inside it takes the next link creation index (several added in one commit take consecutive ones, in the order they are placed), a deletion leaves a gap without lowering the group's counter, and copying the group keeps the order it had. A group *created* that way is a plain group and does not itself track the order, since the tracking is a property of the group whose links are being numbered. Storage transitions preserve the tracking and indexing policy, per-link indexes, and counter. Dense groups have a creation-order index when their policy enables indexing.

[`File::commit`](crate::File::commit) selects link storage from the final count after all staged additions and deletions. Compact groups become dense above the Group Info maximum, and dense groups become compact below its minimum. The defaults are eight and six links. Root and nested groups follow the same rules. Supported hard, soft, and external targets and name encodings are preserved. With persistent free-space management, superseded link heaps and indexes become reusable after publication when their complete ownership is established.

Rebuilding a version 2 object header preserves the two optional blocks its prefix may carry: the four **timestamps** (access, modification, change, birth) that libhdf5 stores on every header it writes, and the **attribute phase-change thresholds** set with `H5Pset_attr_phase_change` ([#422](https://github.com/CramBL/hdf5-pure/pull/422)). The modification and change times are moved to the time of the edit, since a rewrite is a modification. The access and birth times are copied as they were, and a `no_std` build, having no clock, copies all four. A header this crate creates from nothing leaves the timestamp block out, so its whole-file output stays byte-for-byte reproducible. The thresholds are preserved and left unread: the editor's own compact-to-dense switch is the fixed eight attributes that [Dense attribute storage](crate#dense-attribute-storage) describes.

[`copy`](crate::File::copy) performs a deep copy: fresh copies of every object's data and header are written, internal links and the contiguous data address are repointed to the copies, and a link named by the last component of `dst` is added to its parent group. The original is untouched. `src` must exist and `dst` must not (and may not lie inside `src`). Compact attributes are carried over byte-for-byte, including the latest-format form the C library and h5py write, where a handful of inline attributes are accompanied by an Attribute Info message. Dense (fractal-heap) attribute storage, which an object takes on above 8 attributes or when a single attribute is too large for an object-header message, is also reproduced: the source attributes are read out of the source heap and re-emitted into a fresh single-direct-block fractal heap plus B-tree v2 name index in the destination, with a creation-order index beside it where the source object declared one. An attribute too large even for a managed heap object is re-emitted as a *huge* object, as it was in the source. More attributes than the single B-tree leaf can index is rejected, and the attribute is named in the error, never mis-encoded. The set's *total* size has no limit, since the heap's one block is sized to the content.

[`copy_from`](crate::File::copy_from) is the same operation **across two open files**, the cross-file form of `H5Ocopy`. The source lives in a separate [`File`](crate::File) reader, apart from the file being edited:

```rust
# let dir = tempfile::tempdir()?;
# let library_path = dir.path().join("library.h5");
# let path = dir.path().join("output.h5");
# let mut source = hdf5_pure::FileBuilder::new();
# source.create_dataset("calibration").with_f64_data(&[0.5, 1.5]);
# source.write(&library_path)?;
# let mut builder = hdf5_pure::FileBuilder::new();
# let mut run2 = builder.create_group("run2");
# run2.create_dataset("signal").with_f64_data(&[1.0]);
# builder.add_group(run2.finish());
# builder.write(&path)?;
use hdf5_pure::File;

let library = File::open(&library_path)?;
let file = File::open_rw(&path)?;
file.copy_from(&library, "calibration", "run2/calibration")?;
file.commit()?;
# drop(file);
# assert_eq!(File::open(&path)?.dataset("run2/calibration")?.read_f64()?, vec![0.5, 1.5]);
# Ok::<(), hdf5_pure::Error>(())
```

Unlike [`copy`](crate::File::copy), the source subtree is read and validated **eagerly** (the [`File`](crate::File) borrow need not outlive the call), so [`copy_from`](crate::File::copy_from) returns a `Result`, and the destination still changes only on [`commit()`](crate::File::commit). Because the copy is byte-for-byte verbatim, anything whose stored bytes embed a *source-file* absolute address, which would dangle in another file, is rejected up front: variable-length and reference datasets and attributes (whether compact or dense), and any shared header message (a committed datatype, or an SOHM-shared dataspace, fill value, or filter pipeline). The same-file [`copy`](crate::File::copy) keeps these forms valid instead, by sharing the source file's global heaps and objects. The `source` must be a buffered file ([`File::open`](crate::File::open) or [`File::from_bytes`](crate::File::from_bytes), and not [`File::open_streaming`](crate::File::open_streaming)) using 8-byte offsets and no userblock.

### Replacing an object

Deleting a path and creating something new at that same path in one commit replaces it. The removal is applied before the addition, and the commit's single superblock write publishes both, so a rotating store, a ring buffer of tables or a rolling window of daily datasets, expresses a rotation as one commit, and the path is never momentarily absent:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("ring.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("t0").with_i32_data(&[0]);
# builder.write(&path)?;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
let root = file.root();

root.delete("t0")?;
root.create_dataset("t0", |b| { b.with_i32_data(&[1, 2, 3]); })?;
file.commit()?;
# drop(root);
# drop(file);
# assert_eq!(File::open(&path)?.dataset("t0")?.read_i32()?, vec![1, 2, 3]);
# Ok::<(), hdf5_pure::Error>(())
```

The replacement need not resemble the original: a dataset may replace a group or the reverse, and replacing a group discards its whole subtree.

Everything the commit adds *below* a replaced path lands in the replacement, whatever order the calls were made in: staging [`create_dataset("g/x")`](crate::Group::create_dataset) and then replacing `g` puts `x` in the new group, never in the one being removed. What is rejected is a staged edit that could only mean the *original*:

| staged beside [`delete("g")`](crate::Group::delete) | |
| --- | --- |
| [`create_dataset("g/x")`](crate::Group::create_dataset) with no replacement of `g` | rejected: it would add into a group whose own link the commit removes |
| a group under `g` that the commit does not itself create | rejected |
| an attribute set on `g`, or a value overwrite inside it | rejected |
| [`copy("g/inner", "backup")`](crate::File::copy), a copy reading from the replaced subtree | rejected: a copy takes its bytes from the pre-commit file, so this would place the original at `backup` while the replacement lands at `g` |

A source that is deleted but *not* replaced is unambiguous, since that is a move, and stays allowed.

The new object's storage is appended, and the original's bytes stay where they are until the superblock is repointed, which is what makes a crash mid-rotation land on one side or the other. The space the deletion released is therefore what a *later* commit draws on, by the two mechanisms [Space reuse and truncation](#space-reuse-and-truncation) describes: reuse for an interior region, truncation for one that reaches the end of the file. A rotation loop in one session reaches a steady file size.

## Staged objects are addressable straight away

[`create_group`](crate::Group::create_group), [`create_group_with`](crate::Group::create_group_with) and [`create_dataset`](crate::Group::create_dataset) each return a handle onto the object they stage, and [`File::group`](crate::File::group) / [`File::dataset`](crate::File::dataset) / [`Group::group`](crate::Group::group) / [`Group::dataset`](crate::Group::dataset) find it by name from the same session. A nested schema is therefore built and its handles cached in one pass, with no commit in the middle:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("runs.h5");
# hdf5_pure::FileBuilder::new().write(&path)?;
use hdf5_pure::{File, MaxExtent};

let file = File::open_rw(&path)?;
let epoch = file.root().create_group_with("epoch", |g| {
    g.create_dataset("sys_time", |b| {
        b.with_u64_data(&[])
            .with_shape(&[0])
            .with_maxshape(&[MaxExtent::Unlimited])
            .with_chunks(&[512]);
    });
})?;

let mut sys_time = epoch.dataset("sys_time")?; // before the commit
assert_eq!(sys_time.shape()?, vec![0]);
sys_time.append_staged(|a| { a.append_u64(&[1, 2, 3]); })?;

file.commit()?;
assert_eq!(sys_time.read_u64()?, vec![1, 2, 3]); // the same handle
# Ok::<(), hdf5_pure::Error>(())
```

Such a handle *addresses* the object: it cannot read bytes the file has yet to hold. What it does before the commit:

| Through a handle onto a staged object | Before [`commit`](crate::File::commit) |
| --- | --- |
| [`Group::create_group`](crate::Group::create_group) / [`create_group_with`](crate::Group::create_group_with) / [`create_dataset`](crate::Group::create_dataset) / [`delete`](crate::Group::delete) / [`set_attr`](crate::Group::set_attr) / [`remove_attr`](crate::Group::remove_attr) | stages, exactly as on a committed group |
| [`Group::group`](crate::Group::group) / [`dataset`](crate::Group::dataset), [`groups()`](crate::Group::groups) / [`datasets()`](crate::Group::datasets) / [`iter_groups()`](crate::Group::iter_groups) / [`iter_datasets()`](crate::Group::iter_datasets) | the staged members |
| [`Dataset::shape`](crate::Dataset::shape) / [`maxshape`](crate::Dataset::maxshape) / [`dtype`](crate::Dataset::dtype) / [`datatype`](crate::Dataset::datatype) / [`is_chunked`](crate::Dataset::is_chunked) / [`filters`](crate::Dataset::filters) | what the builder settled |
| [`Dataset::append_staged`](crate::Dataset::append_staged) | the elements join the pending creation |
| [`Dataset`](crate::Dataset) reads (`read_*`, [`attrs`](crate::Dataset::attrs)), [`Group::attrs`](crate::Group::attrs) | [`Error::NotCommitted`](crate::Error::NotCommitted) |
| [`Dataset::append`](crate::Dataset::append) / [`append_raw`](crate::Dataset::append_raw) / [`buffered_appender`](crate::Dataset::buffered_appender) | [`Error::NotCommitted`](crate::Error::NotCommitted): no bytes to grow in place |
| [`Dataset::write`](crate::Dataset::write) / [`write_staged`](crate::Dataset::write_staged) / [`set_attr`](crate::Dataset::set_attr) / [`remove_attr`](crate::Dataset::remove_attr) | [`Error::NotCommitted`](crate::Error::NotCommitted) |

[`append_staged`](crate::Dataset::append_staged) on a staged dataset is the same edit as having handed the elements to its builder, so it needs neither an unlimited dimension nor a chunked layout, the two an append onto a *committed* dataset does need. A variable-length-string or object-reference dataset is the exception: its per-element side table is built with the dataset, so supply every element through the builder.

After [`commit`](crate::File::commit) the same handles read and write the objects in the file. A rejected commit leaves them staged and still returning [`NotCommitted`](crate::Error::NotCommitted), and the same batch can be committed again.

These handles are ordinary [`Group`](crate::Group) and [`Dataset`](crate::Dataset) handles, so each **holds the file's exclusive OS lock alive**: `let ds = root.create_dataset(..)?` holds the lock for as long as `ds` lives, where `root.create_dataset(..)?;` drops it at the semicolon. Reopening the file (mandatory on Windows, where the lock is not advisory) fails until the handle is dropped. See [`File::close`](crate::File::close) for the rule.

Only a path the session **named** is addressable this way, and every group on the way to an addition has to be in the file already or staged in the same session. [`create_dataset("a/b")`](crate::Group::create_dataset) with no `a` anywhere stages, and the commit rejects the whole batch with [`Error::EditUnsupported`](crate::Error::EditUnsupported) ([#533](https://github.com/CramBL/hdf5-pure/issues/533)). Stage the group by name ([`create_group("a")`](crate::Group::create_group)) before the dataset, which gives the commit its parent and hands you a handle on it.

Two more rules follow from what [`commit`](crate::File::commit) itself does. A creation whose name **collides** with a link the commit keeps is rejected *where it is staged*, before the commit runs, because there would be no new object for the handle to address: [`create_dataset("existing")`](crate::Group::create_dataset) returns [`Error::EditUnsupported`](crate::Error::EditUnsupported) unless the same session also deletes that link. A second creation at a path the session **already staged** one at is rejected there too, since one name cannot mean two objects: the staged set is indexed by path, so the handle would report the first creation's shape and datatype as the second one's. Staging the same *group* twice is the exception and stays allowed, since both calls refer to one group, which is how attributes and children are added to a group already staged. And deleting something the session staged and has not committed **withdraws** that staging, its attributes, appends and staged children with it, since there is no link in the file to unlink. Where the file does hold a link at the path, the plain deletion of it is what remains. A handle onto a withdrawn creation reports [`Error::StagingWithdrawn`](crate::Error::StagingWithdrawn) from then on: it refers to nothing, and the object the file holds at that path is the one the session is deleting.

A deletion carries a group's whole subtree away, but only a commit that builds that group **again** puts anything back under it. So [`delete("g")`](crate::Group::delete) plus [`create_dataset("g/x")`](crate::Group::create_dataset) with no new `g` is a batch the commit rejects, and until then `g`'s own children still own their names, so reads and listings of them keep working. The root cannot be deleted.

[Replacing an object](#replacing-an-object) reads the same before the commit as after: the path refers to the replacement, and not to the object being removed. A deletion on its own hides nothing, since the object is in the file until the commit runs, and reads through it keep working.

[`StagedGroup`](crate::StagedGroup), which [`create_group_with`](crate::Group::create_group_with)'s closure receives, describes a whole subtree in one call, and a group that is not committed is reachable by name as well. Its methods return `&mut Self` for chaining and hand back no handles, so look the object up by name once the closure has returned.

## Appending to an unlimited dataset

[`Dataset::append_staged`](crate::Dataset::append_staged) grows an existing **chunked, unlimited** dataset in place along its first (axis-0) dimension, **including filtered** datasets (deflate, shuffle, fletcher32, scale-offset, LZF, and ZFP with the `zfp` feature). It is the general, non-SWMR counterpart to the SWMR writer, which appends only to *unfiltered*, chunk-aligned datasets. It returns an [`AppendBuilder`](crate::AppendBuilder) whose typed and generic methods mirror the writer's, and repeated calls concatenate in call order.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_i32_data(&[5, 6, 7])
#     .with_maxshape(&[MaxExtent::Unlimited])
#     .with_chunks(&[4]);
# builder.write(&path)?;
# use hdf5_pure::MaxExtent;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
file.dataset("samples")?
    .append_staged(|a| {
        a.append_i32(&[8, 9, 10, 11]);
    })?;
file.commit()?;
# drop(file);
# assert_eq!(File::open(&path)?.dataset("samples")?.read_i32()?, vec![5, 6, 7, 8, 9, 10, 11]);
# Ok::<(), hdf5_pure::Error>(())
```

Existing chunks stay exactly where they are. Only the newly appended chunks are compressed and written, plus the single trailing partial chunk, when the dataset's current length is not a whole multiple of the chunk length. Every other chunk is carried into the rebuilt index by metadata alone, so an append does not rewrite existing data and the file does not grow by the whole dataset each time. Appends of any length are allowed, and the datatype, fill value, filter pipeline, and attributes are preserved.

Like every staged edit, an append commits by writing the new chunks and a rebuilt index at end-of-file and repointing the superblock last (under the file's exclusive lock), so a crash leaves either the original dataset or the fully grown one, never a torn state. It sets no SWMR flag.

### Eligibility

The engine supports the Extensible-Array chunk index, which libhdf5 and h5py select for a single unlimited dimension under the latest format and this crate writes for every unlimited dataset, on rank-1 datasets that have a single hard link. A dataset that is not chunked, not unlimited along axis 0, not Extensible-Array indexed, higher than rank 1, uses a filter this engine cannot re-encode, or has a sparse chunk grid is rejected with [`Error::AppendUnsupported`](crate::Error::AppendUnsupported) before any file bytes change. Check eligibility up front with the read-side accessors [`is_chunked`](crate::Dataset::is_chunked), [`maxshape`](crate::Dataset::maxshape), [`chunk_shape`](crate::Dataset::chunk_shape), and [`filters`](crate::Dataset::filters), which report it without an attempted append (the [reading guide](crate::_guide::reading#chunking-filters-and-append-eligibility) has the four in one table).

Element types are checked, never coerced: each typed `append_*` call records the datatype it implies, and [`commit`](crate::File::commit) rejects a mismatch against the dataset's on-disk datatype, a mix of element types in one builder included. [`append_raw`](crate::Dataset::append_raw) appends already-little-endian element bytes verbatim. Its length must be a whole multiple of the element size. One-byte fixed-point leaves (`u8`/`i8`) are byte-order agnostic, all other numeric leaves must be little-endian. Variable-length and reference leaves are not raw-appendable.

This section mirrors [`examples/append_dataset.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/append_dataset.rs). Run it with `cargo run --example append_dataset`.

### Streaming appends

[`append_staged`](crate::Dataset::append_staged) rebuilds the dataset's chunk index and relocates its header on every [`commit`](crate::File::commit) (and each new [`File::open_rw`](crate::File::open_rw) re-reads the metadata it needs, or the whole file when it falls back to the mirror), which is the right trade for a one-off append composed alongside other edits, and the wrong one for a high-frequency append loop. For that, open the file **once** with [`File::open_rw`](crate::File::open_rw) and append many times through a [`Dataset`](crate::Dataset) handle, growing the Extensible-Array index *in place*: each append then costs `O(appended bytes)` plus amortized `O(1)` index overhead, with no whole-file re-read and no index rebuild.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_i32_data(&[5, 6, 7])
#     .with_maxshape(&[MaxExtent::Unlimited])
#     .with_chunks(&[4]);
# builder.write(&path)?;
# use hdf5_pure::MaxExtent;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
let mut samples = file.dataset("samples")?;
samples.append(&[8i32, 9, 10, 11])?;
samples.append(&[12i32, 13])?; // any length, from any length
file.close()?;
# drop(samples);
# assert_eq!(File::open(&path)?.dataset("samples")?.read_i32()?.len(), 9);
# Ok::<(), hdf5_pure::Error>(())
```

The caller addresses datasets by name through one open file. With ordinary write buffering, the
writer orders payloads before their references and publishes the new dataspace dimension after the
index updates. The writer preserves publication order under either [`SyncPolicy`](crate::SyncPolicy)
and forces durability at each barrier under the default policy.
[Choosing the fsync cadence](#choosing-the-fsync-cadence) describes durability timing.
Explicit [page buffering](#letting-pages-live-longer) has separate interruption guarantees.

An append may start at an unaligned length under a lossless pipeline. The writer decodes the partial
tail, extends it, and places the replacement separately from its predecessor. It publishes the new
index record after placing the payload and recording the EOF needed to cover it. After confirming
the index publication, the editor retires the predecessor's complete allocation: the full chunk
size for unfiltered storage, or the index record's actual stored size for filtered storage, including
filter-exempt payloads. Under a lossless pipeline, the live prefix is preserved.

A partial tail stored under a lossy pipeline (ZFP or float D-scale scale-offset) is rejected by
default, because re-encoding it can change values already published in that tail. Under
[`PartialEdgeFiltering::Skip`](crate::PartialEdgeFiltering::Skip), the writer stores partial tails
unfiltered and permits them to grow. With a lossy pipeline, completing an exempt tail requires two
reference replacements. The editor retires the original allocation after confirming its raw
replacement, then retires the intermediate raw allocation after confirming the filtered record
published after the new extent.

[`Dataset::append_with_options`](crate::Dataset::append_with_options) and its raw and staged counterparts accept [`AppendOptions`](crate::AppendOptions). Setting [`LossyTailReencoding::Allow`](crate::LossyTailReencoding::Allow) through [`with_lossy_tail_reencoding`](crate::AppendOptions::with_lossy_tail_reencoding) permits rewriting a filtered lossy partial tail. Each rewrite can introduce additional quantization error in previously published values. The options apply to the supplied append, or to every write during a buffered appender's lifetime. They are not stored in the file, so reopening the file does not change the default.

An immediate append with that permission can publish the replacement tail's index record before its new length. If interrupted there, a reader at the old length can observe re-quantized values. The file remains readable, but preserving old tail values and publishing the length is not one atomic operation. Complete chunks preceding the tail retain their storage.

The appended **length** is unconstrained too. An unaligned length only makes the last chunk the append writes a partial one, and that chunk's index element is a fresh insert past the old dimension, invisible until the dimension is published, exactly like every whole chunk beside it.

The SWMR writer is the exception: it keeps the chunk-aligned, unfiltered subset, because its readers are concurrent by contract and follow the chunk-index counts and not the dataspace dimension.

The remaining eligibility rules match [`Dataset::append_staged`](crate::Dataset::append_staged)
(chunked, unlimited axis 0, Extensible-Array index, rank 1, a re-encodable filter pipeline), plus the
file-level gates in [the tables above](#choosing-a-write-path). The index must already be allocated
for immediate growth. The writer allocates it eagerly, so an empty dataset it wrote can be grown
from the first append. For an empty dataset created with a deferred index, make the first append
with [`Dataset::append_staged`](crate::Dataset::append_staged), which materializes the index, or
create the dataset with initial data.

Eligible retired tails are reusable during the session. Admission and persistence follow the
[space-reuse rules](#space-reuse-and-truncation), including strategy, threshold, and PAGE class
requirements. The editor consumes retirement records after confirmed replacements and coalesces
eligible ranges into allocator state.

This section mirrors [`examples/append_streaming.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/append_streaming.rs). Run it with `cargo run --example append_streaming`.

### Buffered appends

[`Dataset::append`](crate::Dataset::append) writes each batch immediately, with barriers governed by [`SyncPolicy`](crate::SyncPolicy). A caller appending fewer elements than a chunk holds can buffer them to reduce file writes.

[`Dataset::buffered_appender`](crate::Dataset::buffered_appender) returns a [`BufferedAppender`](crate::BufferedAppender) that holds appended elements in memory and writes them only when they complete a chunk. It is this crate's equivalent of libhdf5's raw-data chunk cache, and it carries the same bargain: buffered elements are not in the file until the appender flushes.

[`Dataset::into_buffered_appender`](crate::Dataset::into_buffered_appender) consumes the dataset and returns an [`OwnedBufferedAppender`](crate::OwnedBufferedAppender). It can be stored in a struct without a lifetime parameter and retains the editing session after the original [`File`](crate::File) is dropped. Both ownership forms use the same buffering and recovery rules.

[`Dataset::buffered_appender_with_options`](crate::Dataset::buffered_appender_with_options) and its [consuming counterpart](crate::Dataset::into_buffered_appender_with_options) use the supplied policy for automatic writes, partial flushes, finish, and drop. With [`LossyTailReencoding::Allow`](crate::LossyTailReencoding::Allow), an appender can resume a filtered lossy partial tail and repeatedly publish partial chunks. That policy applies for the appender's lifetime and does not change other append calls, including calls made after reopening the file.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("telemetry.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_f64_data(&[])
#     .with_shape(&[0])
#     .with_maxshape(&[MaxExtent::Unlimited])
#     .with_chunks(&[1024]);
# builder.write(&path)?;
# use hdf5_pure::MaxExtent;
use hdf5_pure::File;

let file = File::open_rw(&path)?;
let mut samples = file.dataset("samples")?;
let mut appender = samples.buffered_appender()?;
for batch in 0..1000 {
    appender.append(&[f64::from(batch); 100])?; // buffered; writes once a chunk fills
}
appender.finish()?;                             // the partial tail reaches the file here
# drop(samples);
# drop(file);
# assert_eq!(File::open(&path)?.dataset("samples")?.shape()?, vec![100_000]);
# Ok::<(), hdf5_pure::Error>(())
```

Automatic writes end at a chunk boundary and leave the remainder buffered. Each write uses the immediate append path, including its quantization and interruption behavior when lossy-tail re-encoding is permitted.

Elements kept only in the buffer are lost on a crash. [`flush`](crate::BufferedAppender::flush) publishes the buffered tail and retains recovery access. [`finish`](crate::BufferedAppender::finish) flushes and consumes the appender, including on error. [`discard`](crate::BufferedAppender::discard) consumes it and returns the pending bytes without writing them. Dropping attempts a terminal flush unless a write has failed, and ignores its error. These operations have the same behavior for both ownership forms.

During automatic writes, partial flushes, and terminal flushes, the editor applies the immediate
append's allocation retirement contract. Manager publication and synchronization follow the
[space-reuse rules](#space-reuse-and-truncation) for both ownership forms.

By default, the writer rejects an existing filtered lossy partial tail during construction. A
default partial flush that would leave such a tail is rejected with its buffer retained. A partial
flush permitted by the selected policy can leave a tail for the next write to rewrite. Under
[`PartialEdgeFiltering::Skip`](crate::PartialEdgeFiltering::Skip), partial tails stay unfiltered until
complete.

The session rejects staged edits to a live appender's dataset or an ancestor, and a second appender on the same dataset, with [`Error::EditUnsupported`](crate::Error::EditUnsupported). For datasets opened by path, unrelated edits and commits are allowed. Appender writes leave those edits pending, and unrelated commits preserve the buffer and claim. An appender over a dataset handle reached by object reference blocks all staged edits and other appenders.

[`File::close`](crate::File::close) closes the shared session without flushing live appender buffers. Appending, flushing, and finishing after closure return [`Error::FileClosed`](crate::Error::FileClosed), including with empty input or an empty buffer. Previously buffered bytes remain available through [`unwritten`](crate::BufferedAppender::unwritten) or [`discard`](crate::BufferedAppender::discard).

A **SWMR** writer requires the appended length to be chunk-aligned as well, so it can never write a partial trailing chunk. [`buffered_appender`](crate::Dataset::buffered_appender) rejects such a session outright, and never takes elements it could not flush.

Memory follows the buffer and not the file: an appender holds the unwritten elements plus a copy of the prefix it is writing, so peak is about twice the chunk size above whatever one call hands it. That is a different bound from the one [`Dataset::append`](crate::Dataset::append) gives ([Bounded-memory appends](#bounded-memory-appends)), which is independent of how much a single call appends.

This section mirrors [`examples/append_buffered.rs`](https://github.com/CramBL/hdf5-pure/blob/main/examples/append_buffered.rs). Run it with `cargo run --example append_buffered`.

### Bounded-memory appends

Appending to a large file needs no special entry point: [`File::open_rw`](crate::File::open_rw) already edits a latest-format file bounded, the read-write sibling of [`open_streaming`](crate::File::open_streaming). Reads are served by positioned I/O with the streaming backend's capabilities, and [`Dataset::append`](crate::Dataset::append) runs the same crash-atomic in-place engine as the mirror, reading and patching only bounded windows (the object header, the extensible-array blocks it touches, and the trailing chunk). A large append is applied in whole-chunk batches, each crash-atomic on its own, so peak memory stays at the configured caches plus a few chunks, independent of the file size and of how much one call appends.

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("huge-log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_i32_data(&[5, 6, 7])
#     .with_maxshape(&[MaxExtent::Unlimited])
#     .with_chunks(&[4]);
# builder.write(&path)?;
# use hdf5_pure::MaxExtent;
use hdf5_pure::{EditBacking, File, FileAccessProperties, MemoryStrategy};

// Bounded because the file allows it; add the hint to make it a requirement
// rather than a preference.
let file = File::open_rw_with_options(
    &path,
    FileAccessProperties::new().with_memory_strategy(MemoryStrategy::Bounded),
)?;
assert_eq!(file.edit_backing(), Some(EditBacking::Bounded));
let mut samples = file.dataset("samples")?;
samples.append(&[8i32, 9, 10, 11])?;
file.close()?;
# drop(samples);
# assert_eq!(File::open(&path)?.dataset("samples")?.read_i32()?.len(), 7);
# Ok::<(), hdf5_pure::Error>(())
```

Bounded editing **does** grow a file that persists its free space, a genuine paged file (`H5F_FSPACE_STRATEGY_PAGE`) included, seeding the on-disk free-space managers on open and rewriting them at [`File::close`](crate::File::close), with paged appends kept page-homogeneous (raw and metadata in separate pages). The [file-space strategy](crate::_guide::file_space) guide walks the paged details. Memory budgets are set with the same [`FileAccessProperties`](crate::FileAccessProperties) as the streaming reader, and cached metadata windows touched by an append are invalidated automatically, so reads through the same file never observe stale bytes.

The metadata cache is **off** by default, which is why a bounded session's resident memory starts near zero and not near the file's size. [`FileAccessProperties::with_metadata_cache(MetadataCacheConfig::new(bytes))`](crate::FileAccessProperties::with_metadata_cache) turns it on to pin the metadata a long append loop keeps re-reading, against a byte budget you choose and that never scales with the file's length. [`File::metadata_cache_stats`](crate::File::metadata_cache_stats) reports the hit rate and occupancy that say whether the budget was the right one.

## How it works

[`commit()`](crate::File::commit) appends each new dataset (its data blob and object header) and each new group, then appends rewritten object headers for every touched group and its ancestors up to the root (omitting any deleted links), and finally repoints the superblock at the new root.

With persistent [`FsmAggr`](crate::FileSpaceStrategy::FsmAggr), the editor can reuse the original
root-header allocation when it is below the free-section threshold and both headers occupy one
chunk of the same size. It first publishes the replacement root, then copies that header into
the reserved original extent and publishes its location. This requires a tree proved free of
object references and an original root without an incoming hard link. Other object headers
follow ordinary threshold admission.

Without an explicit page buffer, the editor writes and `fsync`s each replacement before publishing
its address in the superblock. A failure before the first publication leaves the previous root
readable. During root-header reclamation, the committed replacement root remains readable while
the editor copies it into the reserved original extent. Under
[`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose), the editor orders writes against process
failures and defers synchronization until close. See
[Choosing the fsync cadence](#choosing-the-fsync-cadence).

Staged commits can reorder writes across publication barriers when using an
[explicit page buffer](#letting-pages-live-longer). Immediate appends drain pending writes at
allocation and reference publication boundaries.

**Edit validation runs before the first content write.** A validation error leaves the file on disk untouched. In a non-persistent PAGE session recovering from a publication error, the editor first drains pending writes from the preceding attempt.

A rejection costs the session nothing it had staged, so nothing is lost by attempting a commit. A rejected [`commit()`](crate::File::commit) puts the staged set back whole: [`has_staged_edits()`](crate::File::has_staged_edits) still returns `true`, and committing again returns the same error, having applied no part of the batch. A staging call that returns an error stages nothing, a [`create_group_with`](crate::Group::create_group_with) whose closure had already recorded several edits included.

Which errors arrive where follows from that: a dataset is validated as it is staged, so a bad shape, a missing datatype or an unsupported combination is reported by [`create_dataset`](crate::Group::create_dataset) or [`write_staged`](crate::Dataset::write_staged) itself, while anything that has to read the file is reported by [`commit()`](crate::File::commit): a value overwrite that is not the on-disk datatype or shape, a deletion that overlaps another edit, a target that does not exist.

A value overwrite replaces element bytes and nothing else, so a builder that sets more than that, chunking, filters, an extensible shape, an attribute, a fill value, or a fill time, is rejected by the staging call. [`with_vlen_strings`](crate::DatasetBuilder::with_vlen_strings) is not one of them: overwriting a variable-length-string dataset places a fresh global heap collection and resolves the staged element references against it, on a contiguous, compact, or chunked (including filtered) dataset alike.

Overwriting the same dataset again reclaims the collection the previous overwrite placed, so rotating a dataset's strings within a session reaches a steady file size. An overwrite reclaims only this session's own placements, and only while nothing else can reach them: an in-file [`copy`](crate::File::copy) of a variable-length dataset re-emits its element references verbatim, so the copy refers to the same collections the source does. Any copy, or any raw-bytes write ([`with_raw_data`](crate::DatasetBuilder::with_raw_data)) over a datatype that reaches a heap address, gives up the reclaim for the rest of the session. Supplying raw bytes also drops whatever staging described the bytes they replace, so the two cannot be combined to slip past that. An overwrite never reclaims the collections a dataset held when the session *opened*, since whatever wrote the file owns them, so [`repack`](crate::repack()) is what recovers those.

[`with_path_references`](crate::DatasetBuilder::with_path_references) is rejected, because its staged element bytes are placeholder addresses only a newly created dataset resolves. That is a rejection of the builder, not of the datatype it produces: a reference dataset is still overwritten by supplying bytes that need no resolving ([`with_reference_data`](crate::DatasetBuilder::with_reference_data), [`with_raw_data`](crate::DatasetBuilder::with_raw_data)). An object reference supplied as an address is screened at [`commit`](crate::File::commit) against the objects that commit deletes and the headers it rewrites elsewhere, the same ground a path target is checked on. [Bounded-memory read-write](crate#bounded-memory-read-write) has the screening rules.

## Choosing the fsync cadence

By default, the editor calls `fsync` at every durability barrier and at
[`File::close`](crate::File::close). Recycling root or manager metadata can require several
publications in one commit, each with barriers before and after the superblock write. A caller
configures the cadence with [`SyncPolicy`](crate::SyncPolicy) through
[`FileAccessProperties`](crate::FileAccessProperties).

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("log.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("samples")
#     .with_f64_data(&[])
#     .with_shape(&[0])
#     .with_maxshape(&[MaxExtent::Unlimited])
#     .with_chunks(&[256]);
# builder.write(&path)?;
# use hdf5_pure::MaxExtent;
use hdf5_pure::{File, FileAccessProperties, SyncPolicy};

let file = File::open_rw_with_options(
    &path,
    FileAccessProperties::new().with_sync_policy(SyncPolicy::OnClose),
)?;

let mut samples = file.dataset("samples")?;
for batch in 0..1000 {
    samples.append(&[f64::from(batch); 64])?;  // no fsync
}
drop(samples);
file.close()?;   // applies staged edits and requests durability
assert_eq!(File::open(&path)?.dataset("samples")?.shape()?, vec![64_000]);
# Ok::<(), hdf5_pure::Error>(())
```

With ordinary write buffering, successful commits and appends issue their writes to the operating
system under either policy. Immediate appends also drain explicit page buffers at publication
boundaries. Staged commits can retain page-buffered writes after returning.
[Write gathering](#write-gathering) describes these boundaries. The final file bytes are the same
under either synchronization policy.

Three things are given up, and they are worth separating:

- **Power-loss ordering.** With the barriers gone, a machine that loses power mid-commit can have the superblock repoint on disk without the data it points at.
- **Deferred write errors.** On a filesystem that allocates late, a write that will fail at writeback still returns success, and the `fsync` is where the `ENOSPC` or `EIO` surfaces. Skip it and a commit the filesystem cannot complete returns `Ok`, with nothing left to report it.
- **Cross-host visibility.** "Another process sees it" is the page cache, so it holds on a local filesystem. Under NFS's close-to-open semantics a client may hold writes until a flush, so a reader on another host, a SWMR reader included, may not see them.

### Why the last one is not optional

[`File::sync()`](crate::File::sync) issues pending writes and synchronizes file content and metadata
to durable storage under every synchronization policy. If an earlier append outcome is uncertain,
synchronization returns [`Error::AllocationPublicationUncertain`](crate::Error::AllocationPublicationUncertain)
without issuing further writes. Further appends and commits return the same error. Close and reopen
the file before editing again. An interrupted file can fail to reopen.

New staged edits remain pending for [`commit()`](crate::File::commit), and elements held by a
[`BufferedAppender`](crate::BufferedAppender) require its [`flush()`](crate::BufferedAppender::flush)
first.

In a non-persistent PAGE session, a successful sync also verifies and adopts pending publication. If recovery changes the root, existing dataset and group handles opened by path resolve their objects again on their next read. Address-only handles return [`Error::StaleHandle`](crate::Error::StaleHandle). An ordinary sync with no root change preserves handle revisions.

Successful [`close()`](crate::File::close) applies staged edits, finalizes free-space managers, and
synchronizes the final writes under either policy. Dropping the last handle discards staged edits
and attempts finalization and synchronization without reporting errors. After an uncertain append
outcome, the editor rejects append, commit, and explicit synchronization requests in that session.

Clearing the SWMR-write flags requires a barrier at teardown. If the clear is lost, plain
opens return [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse) until
[`File::clear_status_flags`](crate::File::clear_status_flags) clears the flags after the writer
has stopped. SWMR readers can follow the flagged file.

The whole-file paths are outside all of this: [`FileBuilder::write`](crate::FileBuilder::write) and [`repack`](crate::repack()) never `fsync` under either policy, since each writes a file and hands it over, with no editing session to hold.

The default is [`SyncPolicy::Always`](crate::SyncPolicy::Always), which is every guarantee described above this section.

## Write gathering

The editor gathers pending byte ranges and combines those within the same page when flushing.
With default buffering, general editor barriers drain the gathered writes. Immediate appends also
use allocation and reference publication boundaries: payloads and new index blocks are issued
before their references, and recorded EOF covers complete allocations before those references
are published. Dataspace growth follows the data and index updates.

Under [`SyncPolicy::Always`](crate::SyncPolicy::Always), the editor synchronizes content before
references and synchronizes file metadata when EOF changes. Under
[`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose), these boundaries issue writes without
requesting durability. The ordering applies to process interruption. Torn physical metadata
writes can produce an unreadable file.

### Letting pages live longer

An explicit page buffer permits staged commit writes to remain buffered across general ordering
barriers and across operations:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("ring.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("t0").with_i32_data(&[0]);
# builder.write(&path)?;
use hdf5_pure::{File, FileAccessProperties, SyncPolicy};

let file = File::open_rw_with_options(
    &path,
    FileAccessProperties::new()
        .with_sync_policy(SyncPolicy::OnClose)
        .with_page_buffer_size(4 << 20),
)?;
# file.close()?;
# Ok::<(), hdf5_pure::Error>(())
```

Immediate appends drain this buffer at their allocation and reference publication boundaries.
A successful append has issued its writes to the operating system. Staged commits can retain
writes until the byte budget is reached, explicit synchronization is requested, or the session
closes. The final file bytes are independent of the buffering budget.

A page buffer requires:

- A budget of at least the file-space page size on PAGE files, or 4 KiB on other files
- Persisted free space on PAGE files
- A version-3 superblock
- [`SyncPolicy::OnClose`](crate::SyncPolicy::OnClose)

The editor rejects incompatible settings with [`Error::EditUnsupported`](crate::Error::EditUnsupported).
SWMR writers reject page buffering. [`File::create_with_options`](crate::File::create_with_options)
rejects an incompatible creation/access pair before writing the file.

### Choosing a budget

The byte budget limits pending ranges. Reaching it drains every range, so a smaller budget can
increase the number of writes within a long append batch. Separate small append calls drain at
their own publication boundaries even when the budget could hold more.

A nonzero page-buffer budget replaces the default gathering budget. Staged commit writes can
remain resident across operations, while successful appends drain pending writes before returning.

### Interrupted page-buffered sessions

Staged commit references can be issued before their content when an explicit page buffer is
flushed in address order. An interruption can leave a root pointing at missing metadata or a
dataset pointing at incomplete values.

The editor marks the superblock for the lifetime of a page-buffered session. It synchronizes the
mark at open and clears it during successful closure. After an interruption, ordinary opens
return [`Error::FileMarkedInUse`](crate::Error::FileMarkedInUse) while the mark remains.
[`File::clear_status_flags`](crate::File::clear_status_flags) permits inspection after the writer
has stopped, without guaranteeing data integrity.

[`WriteMarkPolicy::AllowSnapshot`](crate::WriteMarkPolicy::AllowSnapshot) permits a read-only open
on the caller's assertion that the writer has flushed and the stored bytes form a readable state.
It does not permit a second writer. [`File::sync`](crate::File::sync) or successful closure issues
pending writes and requests durability.

## Supported targets and formats

Contiguous and chunked datasets (with any filter the whole-file writer supports), compact-link groups, and dense-link groups are supported. The editor accepts the following formats:

- Version 0, 1, 2, and 3 superblocks.
- Single- and multi-chunk object headers. A multi-chunk header is collapsed into a single chunk on rewrite.
- A version 0/1 symbol-table group on the edited path is converted to the latest format. Adding and deleting are supported on these older files, and copying a version-1 object is not.

The editor rejects anything it cannot reproduce faithfully, with [`Error::EditUnsupported`](crate::Error::EditUnsupported), and never degrades a file silently:

- A file whose superblock is not located at its base address, a relocated or malformed userblock layout. (A canonical userblock, such as a MATLAB v7.3 `.mat` file's 512-byte userblock, is supported: addresses are read and written relative to the base and the userblock bytes are preserved.)
- Dense link storage with heap filters, huge or tiny heap objects, heap IDs other than seven bytes, or Link messages longer than 4096 bytes. Link names must be valid UTF-8 for the editor's paths.
- Editing a **group with more than one hard link**, or one whose link count cannot be established because the file's links cannot be walked. A commit rebuilds a dirty group's object header at a fresh address and repoints the link it resolved the group through, and every other link would be left pointing at the old header, which the same commit frees. The rejection reaches an edit *below* such a group too, since every ancestor on the edited path is rebuilt. The same rule governs a relocating dataset write, append, or attribute edit. The root group is exempt, since the superblock addresses it and no link does. Deleting a link is unaffected, since it relocates nothing.
- Copying an existing version-1 object.
- Across files ([`copy_from`](crate::File::copy_from)): variable-length or reference datasets and attributes, any shared (committed/SOHM) header message, and a streaming source file, none of which can be reproduced verbatim in another file.
- Editing the attributes of an object whose attribute message is a **shared (SOHM) record**, and moving or removing an object header the file's shared-message index lists. A file that shares messages is read and edited otherwise. What a commit cannot do is change the table's reference count or repoint the index, so it rejects the two edits that would need to. [`repack`](crate::repack()) rewrites such a file with every message stored inline ([#417](https://github.com/CramBL/hdf5-pure/issues/417)).
- Copying a dataset whose elements live in **external files** (`H5Pset_external`), in either direction. It carries the same address-less contiguous layout a never-written dataset does, which is copied as the empty storage it is, so the External Data Files message tells the two apart, where the address cannot ([#336](https://github.com/CramBL/hdf5-pure/issues/336)).

[`Error::EditUnsupported`](crate::Error::EditUnsupported) carries the reason as its payload.

## Object references the file already stores

Non-persistent PAGE commits require a tree proved free of object references. The reference updates described below apply to other editing sessions.

An object reference is an object-header address stored as data, in a dataset's elements or in an attribute's value. A commit rebuilds every dirty object's header at a fresh address and repoints the *link* that resolves to it, which leaves those stored addresses pointing at a header the same commit frees. So a commit's last act, after the superblock repoint that publishes it, is to walk the tree it just published and rewrite every stored address that points at something it moved. Nothing about this is opt-in and no API reports it, and the point is that a reference keeps reaching its object.

Two things decide what it reaches. **Where the bytes are**: a contiguous or compact dataset's elements, and an attribute held in the object header, and none of a chunked dataset's chunks, a dense (fractal-heap) attribute, or one held as a shared (SOHM) record. A version 1 object header is read by a different parser that reaches element data and stops short of an attribute's value. Version 1 is the format libhdf5 and h5py write by default, so this is the common case, and not a legacy corner. And **what the datatype is**: an 8-byte object reference, an array of them, a compound holding one, and any nesting of those two, and none of an object reference wider than 8 bytes, a dataset-region reference, a variable-length reference (how the dimension-scale attribute `DIMENSION_LIST` is encoded), or an enumeration over one.

Everything outside that is left exactly as it was, since rejecting it would ban editing such a file at all, because every commit rebuilds its root group ([#324](https://github.com/CramBL/hdf5-pure/issues/324)).

That argument covers what a file already is, and not what an edit makes it. An attribute edit that would move an attribute holding a *repointable* reference out of the object header, where this walk reaches it, into a heap, where it would not, is **rejected** ([#102](https://github.com/CramBL/hdf5-pure/issues/102)): the reference would otherwise stop being repointable for good, since an object that goes dense stays dense. The rejection costs one edit, and the file stays editable. An attribute whose references this walk never repointed anyway, a `DIMENSION_LIST`, a region reference, one wider than 8 bytes, moves freely, and an object the file already stored densely is rebuilt as it is.

A reference whose target the commit **deletes** is left alone too, and for a different reason: the object is gone, and HDF5 has no rule that turns a reference to it into something else. libhdf5 behaves the same way.

The cost is one pass over the file's objects per commit, so it scales with the object count and not with the file's size. A session that proves a file is free of references skips the pass on every later commit, until it stages something that could introduce one.

## Space reuse and truncation

With persistence enabled under PAGE, or under `FsmAggr`, the space a deletion frees is reused for later writes within a session, so add/delete churn keeps the file bounded. If a freed run reaches the end of the file, the file is truncated to where that run starts, so [`File::file_size()`](crate::File::file_size) reports the shorter length and not a high-water mark. On a paged file the end of allocation stays a whole number of pages, so what comes back is whole pages.

Under [`FileSpaceStrategy::Aggr`](crate::FileSpaceStrategy::Aggr), staged edits and immediate appends use separate metadata and raw aggregates with 2048-byte refill blocks. Unused reservations remain available across commits within the session. A clean close or drop releases unused aggregate space at EOA after publishing the shorter EOA with the current root, unless publication is uncertain. Both `Aggr` and [`FileSpaceStrategy::None`](crate::FileSpaceStrategy::None) discard freed extents. `None` allocates at the end of the file and retains its length after deletion.

Under [`FileSpaceStrategy::FsmAggr`](crate::FileSpaceStrategy::FsmAggr), ordinary read-write sessions
try reusable freed space before the metadata or raw aggregate, with persistence enabled or disabled,
including when the file has no recorded strategy. Abandoned aggregate spare is tracked subject to
the free-section threshold. Active reservations are excluded from persisted free sections. A commit
may surrender both windows when their spare permits releasing a larger dead suffix, or when recycling
sub-threshold persistent manager storage. Persistent close or drop returns unused spare through
replacement managers before truncating, provided publication completed successfully. Reopening starts
with empty aggregate windows. SWMR sessions under `FsmAggr` use no classic aggregates.

Two things stay behind. A file that persists its free space rewrites its free-space managers on every commit and cannot put them where the previous set still is, so a few of those blocks' worth of the run is kept for the commits that follow. A run that would give back less than it keeps is left where it is, since trimming the top off a hole the next write was about to fill costs more than it returns. The result is that the file ends just above its last live allocation, and a delete can still make a file longer: recording free space on disk costs metadata that has to go somewhere, and libhdf5 grows a file to persist its managers too.

With persistence enabled under PAGE, or under `FsmAggr`, everything a commit writes can go into freed space: object headers, contiguous data, a chunked dataset's chunk data and index, and a dense attribute heap. Each needs a region large enough to hold it whole, so a dataset larger than every free region is written past the end of the file, never split across two holes.

Contiguous and chunked datasets (chunk index plus chunk data) and whole group subtrees, with compact or dense link storage, are reclaimed. Version 2 B-tree chunk indexes are reclaimed with their raw chunk allocations. Dense attribute and link storage that becomes dead is reclaimed: its name index, its optional creation-order index, and its fractal heap. On a paged file all of these B-tree blocks remain metadata, while dataset chunks remain raw data. A global heap collection, which holds variable-length data and can be shared between objects, is reclaimed when only the deleted objects refer to it, and every collection stays allocated when the editor cannot read each object that can refer to one. Reclaim is best-effort: storage whose blocks cannot be enumerated exhaustively, such as a filtered fractal heap with a filter this crate does not decode or with huge objects, is left as dead bytes, which never risks freeing a region still in use.

For a fractal heap with a managed-space free-space manager, reclamation includes the manager
header and the complete allocated section list, including reserved capacity. The editor
validates the header and the used section-list bytes, with mandatory checksum checks. The sections
recorded within that manager describe logical heap space and are excluded from file-space
reclamation. Invalid or unsupported manager metadata prevents reclamation of the heap.

Before editing or reclaiming a checksummed fractal-heap direct block, the editor validates the
complete block's checksum, including when the `checksum` feature is disabled. Read-only heap
object lookups may read only a block prefix and the object bytes. Those bounded reads
validate structural fields and object bounds without establishing the complete block's checksum
integrity. In a filtered heap, the whole direct block is read and decoded for a lookup, and the
editor validates the checksum on the decoded bytes.

On a persistent paged file ([`FileSpaceStrategy::Page`](crate::FileSpaceStrategy::Page)) an allocation is served from free space of the page type it is writing, or from a page that is wholly free, since a page with nothing in it has no type to contradict, so reuse cannot make metadata and raw data share a page. Freed space whose page type cannot be established, a whole-file-generic free section recorded by another writer, or a chunk index that writer placed among its metadata, is recorded and never handed out, which costs some space and keeps the page separation.

The editor places persistent extension and manager blocks in reusable free space where they fit,
or appends them when no region is large enough. Under `FsmAggr`, it can recycle a contiguous allocation
for its extension and managers that is smaller than the threshold by publishing a replacement
elsewhere first. It then rewrites the reserved allocation and publishes it again if the metadata fits,
without admitting that allocation as ordinary free space. Larger replacements follow ordinary
allocation and threshold admission. After publication, the editor may move trailing managers into
earlier free space and truncate the file. The [file-space guide](crate::_guide::file_space)
describes the conditions for fixed-width root updates to reach a stable length across sessions.

An immediate [`Dataset::append`](crate::Dataset::append) (and a [`BufferedAppender`](crate::BufferedAppender) flush) draws on free space too. A file that forgets its free space at close spends it directly. A file that persists its free space first removes a batch of space from the on-disk managers in a separate publication. A batch gathers every hole the appended chunk fits in, up to a megabyte of them. Appends spend the reserved batch before aggregate capacity under `FsmAggr`, whose active reservations are also excluded from persisted free sections. A crash can strand the unspent remainder of a batch. The SWMR writer always appends at the end of the file, because its readers may be reading a region the writer freed. A dataset written into a fragmented file may have its chunks placed in several holes. Sequential reads then fetch each chunk separately, where a single run coalesces adjacent ones. [`repack`](crate::repack()) restores a single run.

For persistent files, the editor publishes remaining retirement changes in the free-space managers
during [`File::commit`](crate::File::commit) or clean session closure. A commit with no staged edits
includes this work, even when replacements reused existing storage and left EOF unchanged. The
editor retains confirmed retirements if a later phase or manager cleanup fails. Allocations that
may remain referenced after uncertain publication stay unavailable for reuse, and the editor may
defer manager cleanup. To include pending manager changes in an explicit synchronization, call
[`File::commit`](crate::File::commit) before [`File::sync`](crate::File::sync).

Non-persistent PAGE sessions retain freed extents without reuse or truncation. Under the default strategy, freed space is reused within the open session and forgotten on close. For a file created with `H5Pset_file_space_strategy(persist = true)`, freed space is recorded on disk and survives a reopen, which the [file-space strategy](crate::_guide::file_space) guide sets out. For a guaranteed shrink that rewrites the whole file compact across a reopen, use [`repack`](crate::repack()), and the [repack](crate::_guide::repack) guide has the options.

## Verifying edits

Reopen the file with [`File::open`](crate::File::open) to confirm the edits landed:

```rust
# let dir = tempfile::tempdir()?;
# let path = dir.path().join("output.h5");
# let mut builder = hdf5_pure::FileBuilder::new();
# builder.create_dataset("temperature").with_f64_data(&[22.5, 23.1, 21.8]);
# let mut sensors = builder.create_group("sensors");
# sensors.create_dataset("pressure").with_f32_data(&[101.3]);
# builder.add_group(sensors.finish());
# builder.write(&path)?;
# let editing = File::open_rw(&path)?;
# let editing_root = editing.root();
# editing_root.create_group_with("run2", |g| {
#     g.set_attr("kind", AttrValue::AsciiString("trial".into()));
# })?;
# editing_root.create_dataset("run2/signal", |b| {
#     b.with_f64_data(&[1.0, 2.0, 3.0]);
# })?;
# editing.copy("temperature", "temperature_backup")?;
# editing_root.delete("sensors/pressure")?;
# editing.commit()?;
# drop(editing_root);
# drop(editing);
use hdf5_pure::{AttrValue, Error, File, FormatError};

let file = File::open(&path)?;
let signal = file.dataset("run2/signal")?.read_f64()?;
let backup = file.dataset("temperature_backup")?.read_f64()?;
let run2_attrs = file.group("run2")?.attrs()?;

assert_eq!(signal, vec![1.0, 2.0, 3.0]);
assert_eq!(backup, vec![22.5, 23.1, 21.8]);
assert_eq!(run2_attrs["kind"], AttrValue::AsciiString("trial".into()));

let err = file.dataset("sensors/pressure").unwrap_err();
let Error::Format(FormatError::PathNotFound(missing)) = &err else {
    panic!("expected PathNotFound, got {err:?}");
};
assert_eq!(missing, "sensors/pressure");
# Ok::<(), hdf5_pure::Error>(())
```

For background on the append-and-repoint design, see the [crate documentation](crate).
