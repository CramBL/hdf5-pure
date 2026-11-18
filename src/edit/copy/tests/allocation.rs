use core::num::NonZeroU32;

use rstest::rstest;
use test_util::allocation;

use super::super::CopyTree;

#[rstest]
#[case::small(1, "copy_destination_counts_small")]
#[case::large(8_192, "copy_destination_counts_large")]
fn fresh_copy_counts_allocate_one_u32_per_descendant(#[case] count: usize, #[case] name: &str) {
    let tree = CopyTree {
        descendants: (0..count).map(|_| super::group_node(vec![])).collect(),
        root: super::group_node((0..count).collect()),
    };
    let (counts, measured) = allocation::measure(name, || tree.fresh_counts().unwrap());
    assert_eq!(counts.len(), count);
    assert!(counts.iter().all(|count| count.0 == NonZeroU32::MIN));
    assert_eq!(measured.blocks, 1, "{measured}");
    assert_eq!(
        measured.bytes,
        u64::try_from(count * core::mem::size_of::<u32>()).unwrap(),
        "{measured}"
    );
    assert_eq!(measured.live_bytes, measured.bytes, "{measured}");
    let (control_counts, control) = allocation::measure(&format!("{name}_control"), || {
        let retained = vec![0_u32; count];
        let counts = tree.fresh_counts().unwrap();
        std::hint::black_box(retained);
        counts
    });
    assert_eq!(control_counts.len(), counts.len());
    assert!(
        control_counts
            .iter()
            .all(|count| count.0 == NonZeroU32::MIN)
    );
    assert_eq!(control.blocks, 2, "{control}");
    assert_eq!(control.bytes, measured.bytes * 2, "{control}");
    assert_eq!(control.live_bytes, measured.live_bytes, "{control}");
}
