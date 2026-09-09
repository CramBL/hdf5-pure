# Code style

`cargo fmt` formats the code and `cargo clippy` with warnings denied checks it. `Cargo.toml` lists the style lints the crate mutes, with the reason for each. Everything below is what those two tools do not decide.

The crate implements a binary format that the reference C library defined over three decades, so much of the code is careful bookkeeping over bytes. Read some of it before writing your own.

## Design

- **No C dependencies.** The crate is pure Rust. `byteorder` is the only mandatory dependency, and every other one is optional behind a cargo feature. A new mandatory dependency needs an issue first.
- **A `no_std` core.** Everything below the filesystem API builds on `core` and `alloc`. The `std` feature adds the filesystem and nothing else.
- **Interoperability defines correctness.** A file the crate writes is read by the reference C library, h5py and MATLAB, and a file they write is read by the crate.
- **Faithful or nothing.** An operation produces a result correct to the byte, or an error that says why. [Error handling](#faithful-or-nothing) has the rule.

## Ordering

### Top-down ordering within modules

Within a module, we prefer to order items top-down. This means that items within a module will depend on items defined below them, but not (usually) above them. The public API, with more internal dependencies, will be read (and changed) more often, and putting it closer to the top of the module makes it more accessible.

Usually `const` values will thus go on the bottom of the module (least complex, usually no dependencies of their own), although in larger modules it can make sense to place a `const` directly below the user (especially if there is a single user, or just a few co-located users).

The `#[cfg(test)] mod tests {}` module goes on the very bottom, if present. Module declarations (`mod foo;`) come after the imports and before other items.

Top-down ordering decides where an item sits in a module. The [HDF5 file format specification][spec] decides the order of everything that mirrors it: the fields of a struct that holds an on-disk structure are declared in the order the specification lists them, a parser reads them in that order, a writer emits them in that order, and an enum of message types, datatype classes or filter identifiers lists its variants in the specification's order. So a module that implements a specification section reads top-down from the public type to its helpers, and the public type reads like the specification's table.

[spec]: https://support.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html

### Ordering for a given type

For a given type, we prefer to order items as follows:

1. The type definition (`struct` or `enum`)
2. The inherent `impl` block (that is, not a trait implementation)
3. `impl` blocks for traits, from most specific to least specific. The least specific would be something like a `Debug` or `Clone` `impl`.

### Ordering associated functions within an inherent `impl` block

Here is a guide to how we like to order associated functions:

0. Associated functions (that is, `fn foo() {}` and not `fn foo(&self) {}`)
1. Constructors, starting with the constructor that takes the least arguments
2. Public API that takes a `&mut self`
3. Public API that takes a `&self`
4. Private API that takes a `&mut self`
5. Private API that takes a `&self`
6. `const` values

We usually also practice top-down ordering here. Where these are in conflict, make a choice that you think makes sense. For getters and setters, the order should typically mirror the order of the fields in the type definition.

### Attribute ordering

Order attributes so that documentation appears first, and the attributes with the most effect on the meaning and function of the type appear last. For example:

```rust
/// Doc comment always first
#[cfg(feature-gates)]
#[allow(lint-configuration)]
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Foo;
```

Prefer to write `derive`d traits in alphabetical order.

## Functions

### Split a growing function

Before making a function longer, consider splitting part of it out into a function whose name says what the part does. A reader follows a sequence of named steps more easily than one long body.

### Consider avoiding free-standing functions

If a function's semantics or implementation are strongly dependent on one of its arguments, and the argument is defined in a type within the current crate, prefer using a method on the type. Similarly, if a function is taking multiple arguments that originate from the same common type in all call sites it is a strong candidate for becoming a method on the type.

### Order arguments from most generic to most specific

Put the most generic arguments first (the file, the reader or writer context) and the most specific last (the chunk index, the element offset). Call sites then read the same way from function to function, with the context first.

### Use `impl` where possible

We prefer to use `impl ...` for arguments and return types when there is a single use of the type. Generic type argument bounds add a level of indirection that is harder to read in one pass.

### Avoid type elision for fully qualified function calls

We prefer to write [fully qualified function calls] with types included, and not elided. For example:

```rust
// Incorrect:
<_>::default()

// Correct:
Dataspace::default()
```

[fully qualified function calls]: https://doc.rust-lang.org/beta/reference/expressions/call-expr.html#disambiguating-function-calls

### Validation

Where possible, avoid writing `validate` or `check` type functions that try to check for error conditions based on the state of a populated object. Prefer ["parse, don't validate"](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/) style and try to use the type system to make it impossible for invalid states to be represented.

### Error handling

We use `Result` types pervasively throughout the code to signal error cases. Outside of unit and integration tests we prefer to avoid `unwrap()` and `expect()` calls unless there is a clear invariant which can be locally validated by the structure of the code. If there is such an invariant, we usually add a comment explaining how the invariant is upheld. In other cases, and especially for error cases which can arise from the contents of a file, which could be crafted by an attacker, we always handle the error and return it to the caller.

A file is untrusted input. The fuzz targets and the property tests feed the reader damaged files, and a panic on any of them is a bug (`effect:crash`). This includes slice indexing and arithmetic on a value that came from the file: use `.get()` and the checked conversions in `src/convert.rs`, which turn an out-of-range value into a `FormatError`. Every offset and length read from a file goes through one, so that a value which does not fit a 32-bit `usize` is an error and never a truncation.

- Every error is returned to the caller and handled once. Code that converts an error returns the conversion.
- A failure mode is a variant of `Error` or `FormatError`, with the values a caller needs to act on it as fields. A new failure mode is a new variant, not a string.
- An invariant the type system can express is a type: a value that cannot be zero is a `NonZeroU64`.
- An invariant it cannot express is a `debug_assert!`, or an `assert!` where a violation would write a wrong file. The code panics only on a bug in the code.
- A `Result` that is deliberately dropped is dropped with `.ok()`, which stops compiling if the return type changes. `let _ = f();` and `unwrap_or(())` are not used for that.
- A string in an error message is formatted with `{:?}`, so that a path or a name is quoted and an embedded newline or tab is escaped.

### Faithful or nothing

An operation either produces a result that is correct down to the bytes, or it returns an error that says why. The editor returns `Error::EditUnsupported` for an object it cannot reproduce, `repack` returns `Error::RepackUnsupported` and leaves the destination path absent, and a reader returns a `FormatError` for a structure it does not handle. Silently writing a file that differs, or reading a value that differs, is the one thing the crate must never do. A new feature that cannot cover a case reports it as an error of that kind.

## Expressions

### Avoid single-use bindings

We generally make full use of the expression-oriented nature of Rust. For example, when using iterators we prefer to use `map` and the other adapter methods over `for` loops when possible, and will often avoid variable bindings if a variable is only used once. Naming variables takes cognitive effort, and so does tracking references to bindings in your mind. One metric we like to minimize is the number of mutable bindings in a given scope.

Remember that the overall goal is to make the code easy to understand. Adapters can help with this by eliding boilerplate (like replacing a `None => None` arm with a `map()` call), but they can also make it harder to understand the code. One example is that a chain like `.map().map_err()` might be harder to understand than a `match` statement (since, in this case, both of the arms have a significant transformation).

### Use early `return` and `continue` to reduce nesting

The typed nature of Rust can cause some code to end up at deeply indented levels, which we call "rightward drift". This makes lines shorter, making the code harder to read. To avoid this, try to `return` early for error cases, or `continue` early in a loop to skip an iteration.

### Hoist common expression returns

When writing a `match` or `if` expression that has arms that each share a return type (e.g. `Ok(...)`), hoist the commonality outside the `match`. This helps separate out the important differences and reduces code duplication.

```rust
// Incorrect:
match foo {
    1..10 => Ok(do_one_thing()),
    _ => Ok(do_another()),
}

// Correct:
Ok(match foo {
    1..10 => do_one_thing(),
    _ => do_another(),
})
```

### Determinism

The order a `HashMap` or `HashSet` iterates in differs from run to run. Where the order reaches a file, a checksum or a value a test compares, the collection is a `BTreeMap` or `BTreeSet`.

### Destructuring

A function that reads most of a struct's fields destructures it once: `let Self { a, b, c } = self;`. A field added later is then a compile error at every such site.

## Naming

### Use concise names

We prefer concise names, especially for local variables, but prefer to expand acronyms and abbreviations that are not very well known (e.g. prefer `dataspace` over `dspace`, `attribute` over `attr` in a public name). Extremely common short forms like `addr` and `len` are acceptable.

Where the specification or libhdf5 already has a term for a thing, start from that term, so that a reader with the specification open finds the code and a libhdf5 user finds the API. Then judge it: a C name is often terse because of C's conventions, and that terseness sometimes costs readability. `btree`, `superblock`, `dataspace` and `datatype` come from the specification and read well. `sohm` does not, and the code says `shared_message`. `fheap` is `fractal_heap`. When in doubt, expand.

Avoid adding a suffix for a variable that describes its type, provided that its type is hard to confuse with other types. The precision and conciseness trade-off for variable names also depends on the scope of the binding.

When in doubt, be explicit: `msg_id` over `id`. Be terse where it does not hurt readability: `msg_id` over `message_identifier`. Where a value is a string, be over-explicit about which: `path`, `link_name`, `attr_name`, and never a bare `name` where two kinds of name are in scope.

Avoid negations in names. `!non_blocking` and `disconnected == false` slow every reader down, so prefer `connected` over `disconnected` and `initialized` over `uninitialized`.

### Avoid `get_` prefixes

Per the [API guidelines](https://rust-lang.github.io/api-guidelines/naming.html#getter-names-follow-rust-convention-c-getter), `get_()` prefixes are discouraged.

### Enum variants

When implementing or modifying an `enum` type, list its variants in alphabetical order. It is acceptable to ignore this advice when matching the order imposed by an external source, such as the HDF5 file format specification.

Prefer active verbs for variant names. E.g. `Allow` over `Allowed`, `Forbid` over `Forbidden`. Avoid faux-bools like `Yes` and `No`, and prefer a variant name that describes its state.

### Don't elide generic lifetimes

We prefer not to elide lifetimes when naming types that are generic over lifetimes. Always include a lifetime placeholder (e.g. `<'_>`) to avoid confusion.

## Modules

A module is a file. An inline module (`mod foo { .. }`) has two uses: `#[cfg(test)] mod tests` at the bottom of the file it tests, and a private module that seals a trait, so that the trait's bound cannot be implemented outside the crate. Everything else that would be an inline module is a file.

Integration tests under `tests/` are one binary per entry file, so related test files are bundled: an entry file declares them (`tests/suite/main.rs` holding `mod append;` and `mod swmr;`) and cargo compiles one binary for the group.

## Imports

We use 3 blocks of imports in our Rust files:

1. `core`, `alloc` and `std` imports
2. Imports from external crates
3. Crate-internal imports

This makes it easier to see where a particular import comes from.

Within the import blocks we prefer to separate imports that do not share a parent module. For example,

```rust
// Incorrect
use alloc::{format, vec::Vec};

// Correct
use alloc::format;
use alloc::vec::Vec;
```

A function is never imported by its bare name. It is called through at least one level of its path: `convert::u32_from(value)` after `use crate::convert;`, and never `use crate::convert::u32_from;`. A bare function call then always means a function in the current module, and a qualified one says where it comes from.

We prefer to reference types and traits by an imported symbol name and not by a qualified path. The one exception to this is when the symbol name is overly generic, or easily confused between different crates. In this case we prefer to import the symbol name under an alias, or if the parent module name is short, using a one-level qualified path. E.g. for a crate with a local `Error` type, prefer to `use std::error::Error as StdError`.

## Exports

The crate root is the public API. The modules are private, and every public item is re-exported from `lib.rs`, grouped by the part of the API it belongs to, or lives in a public module that a cargo feature adds (`mat`). A public item has exactly one path, and the module tree behind it is free to change: moving an item between modules is not an API change.

A rename keeps the old name for one minor release as a deprecated type alias (`#[deprecated] pub type OldName = NewName;`), which warns at the use site. A deprecated `pub use` does not warn on the crate's MSRV, so it is not the mechanism.

## Comments

[DOC_STYLE.md](DOC_STYLE.md) has the rules for doc comments, for the comments beside the code, and for citing the specification and libhdf5.

All comments (doc comment or not) should be wrapped to 100 columns.

## Tests

### Assert the exact error

A test of a failure asserts which failure. `is_err()` also passes on the failure the test rules out. Take the error out with `unwrap_err()` or a `let ... else`, and match the variant and its fields:

```rust
let err = builder.finish().unwrap_err();
let Error::Format(FormatError::InvalidChunkGeometry(reason)) = &err else {
    panic!("expected InvalidChunkGeometry, got {err:?}");
};
assert_eq!(reason, "chunk logical byte size exceeds the 4 GiB format limit");
```

`assert!(matches!(err, Error::StaleHandle), "{err:?}")` is the short form for a variant with no field to check. The message prints the error a wrong variant fails with.

An error from the C library follows the same rule: the test asserts the major and minor codes the condition produces, `err.contains_major(MajorErrorCode::Symbol)` and `err.contains_minor(MinorErrorCode::NotFound)`, or the frames in order where the order matters. Where one code covers two conditions, the test asserts what separates them, as `assert_c_absent` in `crates/crosscheck/src/lib.rs` does.

`hdf5::Error::Internal` is a failure in the binding. A test does not accept it in place of an error from the library.

`assert!(r.is_ok())` is the same defect. A test unwraps the value and asserts on it, and asserts the whole value where that is feasible: `assert_eq!(names, vec!["a", "b"])` over `assert!(names.contains(&"a"))`.

### Test helpers

A helper that several test binaries share lives in `crates/test-util`. A helper that one binary uses lives in that binary. A helper that needs the C library lives in `crates/crosscheck/src/lib.rs`, so that nothing else links it.

## Misc

### Numeric literals

Prefer a numeric base that fits with the domain of the value being used. E.g. use hexadecimal for the format's signatures, message type identifiers and flag bits, and decimal for sizes and counts. Use digit grouping to make larger numeric constants easy to read, e.g. use `100_000_000` and not `100000000`.

A value the specification defines is a named constant, never a literal at the use site: a signature, a message type, a flag bit, a version number, a size threshold. The constant's doc comment cites the section that defines it.

### Avoid type aliases

We prefer to avoid type aliases as they hide the type behind them and do not provide additional type safety. Using the [newtype idiom](https://doc.rust-lang.org/rust-by-example/generics/new_types.html) is one alternative when an abstraction boundary justifies the added complexity. The one exception is the deprecated alias that keeps an old name across a rename, described under [Exports](#exports).

### Type exhaustiveness

Public enums should be marked as _either_ `#[non_exhaustive]` or `#[allow(clippy::exhaustive_enums)]`. The latter is suitable for enums that are already complete by definition. For example: `enum ByteOrder { LittleEndian, BigEndian }` is complete. Err on the side of marking something `#[non_exhaustive]`.

The same applies to structs, with the detail that no manual marking is needed for structures with at least one private field.

### `no_std`

The format structures and the in-memory machinery use `core` and `alloc`, and only the filesystem API is behind the `std` feature. New code below the high-level API imports from `core` and `alloc`, and `just portability::default` checks the `no_std`, WASM and bare-metal builds.

### `unsafe`

Avoid it. An `unsafe` block comes with a `// SAFETY:` comment that explains the invariant and what upholds it, and a test that `just soundness::miri` runs.
