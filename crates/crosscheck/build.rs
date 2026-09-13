//! Build script for `hdf5-pure-crosscheck`.
//!
//! Emits linker directives for the system libmatio when the `__matio` test
//! feature is enabled.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Cargo sets `CARGO_FEATURE_<NAME>` for each enabled feature. Bail out
    // when the crosscheck feature is disabled.
    if std::env::var_os("CARGO_FEATURE___MATIO").is_none() {
        return;
    }

    configure_libmatio();
}

/// Emit `cargo:rustc-link-arg-tests` directives for the system libmatio.
///
/// Cargo passes `cargo:rustc-link-lib` only to the package's library target,
/// propagating it through the `rlib` ("The -l flag is only passed to the library
/// target of the package", Cargo book). Integration tests that do not reference
/// items from the library target cause rustc to prune the `rlib`, dropping the
/// linker flags. Passing arguments directly to test targets with
/// `rustc-link-arg-tests` ensures libmatio is linked into the test binary.
///
/// Checks well-known Homebrew and MacPorts locations on macOS, otherwise
/// relies on the default linker search path (sufficient for Linux
/// packages that install `libmatio.so` under `/usr/lib/...`).
fn configure_libmatio() {
    let mut candidates = vec![
        "/opt/homebrew/opt/libmatio/lib".into(), // macOS Apple Silicon Homebrew
        "/usr/local/opt/libmatio/lib".into(),    // macOS Intel Homebrew
        "/opt/local/lib".into(),                 // MacPorts
        "/home/linuxbrew/.linuxbrew/opt/libmatio/lib".into(), // Linux brew default
    ];

    // Catch user-local Linux brew installations
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(format!("{home}/.linuxbrew/opt/libmatio/lib"));
    }

    for dir in candidates {
        if std::path::Path::new(&dir).exists() {
            println!("cargo:rustc-link-arg-tests=-L{dir}");
            // Embed the path in the binary so it can find the library at runtime
            println!("cargo:rustc-link-arg-tests=-Wl,-rpath,{dir}");
        }
    }
    println!("cargo:rustc-link-arg-tests=-lmatio");
}
