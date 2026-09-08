//! Profile what this crate allocates, and where (issue #228).
//!
//! The tests under `tests/allocation_bounds.rs` say whether a bound still holds.
//! This says *which call site* spent the bytes, which is the question you have
//! next when one of them fails, or when you are looking for something to make
//! cheaper in the first place.
//!
//! ```sh
//! cargo run --release --example heap_profile
//! ```
//!
//! It writes `target/heap-profile.html` — one self-contained page, open it in a
//! browser — and prints the heaviest call sites to the terminal. Each phase of
//! the workload is a named region, so the page can answer "what did the *write*
//! allocate" separately from "what did the read allocate", which no stack trace
//! can: the same `Vec::with_capacity` in the same helper is a different cost in
//! each.
//!
//! Release rather than debug is deliberate. A debug build's allocation *counts*
//! are the same, but its call stacks are full of the inlining that did not
//! happen, so the profile reads as a list of iterator adapters.

use hdf5_pure::{File, FileBuilder};

#[global_allocator]
static ALLOC: heapscope::Alloc = heapscope::Alloc::system();

/// 8 MiB of f64 in 4 KiB chunks — enough chunks (2,048) that a per-chunk cost
/// stands out against the constants around it.
const ELEMENTS: usize = 1024 * 1024;
const CHUNK_ELEMS: u64 = 512;
/// A vlen dataset packed into one global heap collection, which is the layout
/// that makes a windowed read walk far more than it keeps.
const LABELS: usize = 32 * 1024;
const LABEL_LEN: usize = 128;
/// A metadata-heavy file: many small datasets in one group, each with an
/// attribute. The phases above allocate in proportion to *data*; these allocate
/// in proportion to the number of *objects*, which is a different path.
const OBJECTS: usize = 1024;
/// Appends of one chunk each onto an unlimited dataset, enough of them that a
/// per-append cost that scales with the dataset stands out from one that does
/// not.
const APPENDS: u64 = 512;

/// `<target>/heap-profile.html`, found from this binary rather than assumed to
/// be `./target`: a `build.target-dir` in the user's cargo config moves it.
fn profile_path() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        // <target>/<profile>/examples/heap_profile
        .and_then(|exe| exe.ancestors().nth(3).map(std::path::Path::to_path_buf))
        .unwrap_or_else(|| std::path::PathBuf::from("target"))
        .join("heap-profile.html")
}

#[expect(
    clippy::panic_in_result_fn,
    reason = "the example checks its own output"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profile_path = profile_path();
    let profile_path = profile_path.display().to_string();
    let profiler = heapscope::Profiler::builder()
        .output(heapscope::Output::html(&profile_path))
        .build()
        .map_err(|e| {
            format!(
                "heapscope could not start: {e}\n\
                 On x86_64 this needs the frame pointers `.cargo/config.toml` \
                 sets for that target; a build that overrides RUSTFLAGS drops them."
            )
        })?;

    let dir = tempfile::tempdir()?;
    let numeric = dir.path().join("numeric.h5");
    let labels = dir.path().join("labels.h5");

    {
        // Built outside the region, as for the strings below: the 8 MiB of input
        // is the caller's, and counting it here would report the writer as
        // holding one more copy than it does.
        let data: Vec<f64> = (0..ELEMENTS).map(|i| i as f64).collect();
        let _region = heapscope::region("write chunked");
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("t")
            .with_f64_data(&data)
            .with_shape(&[ELEMENTS as u64])
            .with_chunks(&[CHUNK_ELEMS]);
        builder.write(&numeric)?;
    }

    {
        // The strings are built outside the region: this profile is of what the
        // crate allocates, and 32k `format!` calls would otherwise be four fifths
        // of this phase's blocks and read as the writer's.
        let strings: Vec<String> = (0..LABELS)
            .map(|i| format!("{i:0>width$}", width = LABEL_LEN))
            .collect();
        let refs: Vec<&str> = strings.iter().map(String::as_str).collect();
        let _region = heapscope::region("write vlen strings");
        let mut builder = FileBuilder::new();
        builder.create_dataset("labels").with_vlen_strings(&refs);
        builder.write(&labels)?;
    }

    let numeric_file = File::open_streaming(&numeric)?;
    let numeric_ds = numeric_file.dataset("t")?;

    let whole = {
        let _region = heapscope::region("read chunked whole");
        numeric_ds.read_raw()?
    };
    assert_eq!(
        whole.len(),
        ELEMENTS * 8,
        "whole read returned wrong length"
    );
    drop(whole);

    let window = {
        let _region = heapscope::region("read chunked window");
        numeric_ds.read_f64_rows(ELEMENTS as u64 / 2, 4096)?
    };
    assert_eq!(window.len(), 4096, "row window returned wrong length");
    assert_eq!(
        window[0].to_bits(),
        ((ELEMENTS / 2) as f64).to_bits(),
        "row window started at the wrong row"
    );
    drop(window);

    let labels_file = File::open_streaming(&labels)?;
    let labels_ds = labels_file.dataset("labels")?;

    let strings = {
        let _region = heapscope::region("read vlen window");
        labels_ds.read_string_rows(LABELS as u64 / 2, 256)?
    };
    assert_eq!(strings.len(), 256, "string window returned wrong length");
    assert_eq!(
        strings[0].trim_start_matches('0'),
        (LABELS / 2).to_string(),
        "string window started at the wrong row"
    );
    drop(strings);

    let typed = {
        let _region = heapscope::region("read chunked whole typed");
        numeric_ds.read_f64()?
    };
    assert_eq!(
        typed.len(),
        ELEMENTS,
        "typed whole read returned wrong length"
    );
    drop(typed);

    let all_strings = {
        let _region = heapscope::region("read vlen whole");
        labels_ds.read_string()?
    };
    assert_eq!(
        all_strings.len(),
        LABELS,
        "whole string read returned wrong length"
    );
    drop(all_strings);

    // --- the metadata paths -------------------------------------------------

    let many = dir.path().join("many.h5");
    {
        let _region = heapscope::region("write many objects");
        let mut builder = FileBuilder::new();
        for i in 0..OBJECTS {
            builder
                .create_dataset(&format!("d{i:05}"))
                .with_i32_data(&[i as i32])
                .set_attr("units", hdf5_pure::AttrValue::I32(i as i32));
        }
        builder.write(&many)?;
    }

    let many_file = {
        let _region = heapscope::region("open many objects");
        File::open_streaming(&many)?
    };

    let names = {
        let _region = heapscope::region("list many objects");
        many_file.root().datasets()?
    };
    assert_eq!(names.len(), OBJECTS, "listing returned the wrong count");

    {
        let _region = heapscope::region("resolve many objects");
        for name in &names {
            let _ds = many_file.dataset(name)?;
        }
    }

    {
        let _region = heapscope::region("read many attributes");
        for name in &names {
            let ds = many_file.dataset(name)?;
            let attrs = ds.attrs()?;
            assert_eq!(attrs.len(), 1, "each dataset carries one attribute");
        }
    }

    // --- editing an existing file ------------------------------------------

    // Appending in place is the write path that runs many times over one file
    // rather than once, so what it costs *per call* is what matters: a cost that
    // scales with the dataset rather than the batch is the defect to look for.
    let growing = dir.path().join("growing.h5");
    {
        let mut builder = FileBuilder::new();
        builder
            .create_dataset("t")
            .with_f64_data(&[0.0f64; CHUNK_ELEMS as usize])
            .with_shape(&[CHUNK_ELEMS])
            .with_maxshape(&[u64::MAX])
            .with_chunks(&[CHUNK_ELEMS]);
        builder.write(&growing)?;
    }

    let batch: Vec<f64> = (0..CHUNK_ELEMS).map(|i| i as f64).collect();
    {
        // One `fsync` at close rather than one per append: the appends are what
        // is being measured, and the barriers between them are not free.
        let file = File::open_rw_with_options(
            &growing,
            hdf5_pure::FileAccessProperties::new().with_sync_policy(hdf5_pure::SyncPolicy::OnClose),
        )?;
        let mut ds = file.dataset("t")?;
        let _region = heapscope::region("append in place");
        for _ in 0..APPENDS {
            ds.append(&batch)?;
        }
    }
    let grown = File::open(&growing)?;
    assert_eq!(
        grown.dataset("t")?.shape()?,
        vec![CHUNK_ELEMS * (APPENDS + 1)],
        "the appends did not all land"
    );
    drop(grown);

    profiler.print_summary(16)?;
    // Dropping the profiler is what writes the page, so the check below has to
    // come after it rather than at the end of `main`.
    drop(profiler);

    let written = std::fs::metadata(&profile_path)
        .map_err(|e| format!("{profile_path} was not written: {e}"))?
        .len();
    assert!(written > 0, "{profile_path} was written empty");
    println!("\nwrote {profile_path} ({written} bytes) — open it in a browser");

    Ok(())
}
