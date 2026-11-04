use core::num::NonZeroU32;

use rstest::rstest;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use test_util::allocation;

use super::{CountArithmeticError, CountBalance};

#[rstest]
#[case::empty(0, 0, 0, 0)]
#[case::unchanged(7, 0, 0, 7)]
#[case::addition(7, 3, 0, 10)]
#[case::removal(7, 0, 3, 4)]
#[case::cancellation(7, 3, 3, 7)]
#[case::zero(3, 0, 3, 0)]
#[case::positive_added(3, 2, 3, 2)]
#[case::zero_transition(1, 0, 1, 0)]
#[case::added_reference_retained(1, 1, 1, 1)]
#[case::maximum_unchanged(u32::MAX, 0, 0, u32::MAX)]
#[case::maximum_cancellation(u32::MAX, 1, 1, u32::MAX)]
#[case::maximum_added(0, u64::from(u32::MAX), 0, u32::MAX)]
#[case::maximum_net(3, u64::from(u32::MAX), 3, u32::MAX)]
#[case::maximum_removed(u32::MAX, u64::from(u32::MAX), u32::MAX, u32::MAX)]
fn count_balances_check_the_completed_net_count(
    #[case] original: u32,
    #[case] added: u64,
    #[case] removed: u32,
    #[case] remaining: u32,
) {
    let mut balance = CountBalance::new(original, added);
    assert_eq!(balance.is_zero(), original == 0 && added == 0);
    if let Some(removed) = NonZeroU32::new(removed) {
        balance.remove(removed).unwrap();
    }
    assert_eq!(balance.is_zero(), remaining == 0);
    assert_eq!(balance.remaining().unwrap(), remaining);
}

#[rstest]
#[case::immediate(u32::MAX, 1, 0)]
#[case::after_removal(u32::MAX, 2, 1)]
#[case::wide_addition(0, u64::from(u32::MAX) + 1, 0)]
#[case::maximum_addition(0, u64::MAX, 0)]
#[case::wide_sum(u32::MAX, u64::MAX, 1)]
fn count_balance_overflows_report_the_complete_original_operands(
    #[case] original: u32,
    #[case] added: u64,
    #[case] removed: u32,
) {
    let mut balance = CountBalance::new(original, added);
    if let Some(removed) = NonZeroU32::new(removed) {
        balance.remove(removed).unwrap();
    }
    assert!(!balance.is_zero());
    assert_eq!(
        balance.remaining().unwrap_err(),
        CountArithmeticError::Overflow {
            original,
            added,
            removed: u64::from(removed),
        }
    );
}

#[rstest]
#[case::zero(0, 0)]
#[case::positive(3, 0)]
#[case::added_reference(3, 1)]
#[case::maximum_addition(0, u64::MAX)]
#[case::maximum_original(u32::MAX, 0)]
fn excess_original_removals_return_underflow(#[case] original: u32, #[case] added: u64) {
    let mut balance = CountBalance::new(original, added);
    if let Some(removed) = NonZeroU32::new(original) {
        balance.remove(removed).unwrap();
    }
    assert_eq!(
        balance.remove(NonZeroU32::MIN).unwrap_err(),
        CountArithmeticError::Underflow {
            original,
            added,
            removed: u64::from(original) + 1,
        }
    );
    assert_eq!(balance.is_zero(), added == 0);
    let remaining = balance.remaining();
    if let Ok(added) = u32::try_from(added) {
        assert_eq!(remaining.unwrap(), added);
    } else {
        assert_eq!(
            remaining.unwrap_err(),
            CountArithmeticError::Overflow {
                original,
                added,
                removed: u64::from(original),
            }
        );
    }
}

#[test]
fn failed_removal_preserves_the_successful_prefix() {
    let mut balance = CountBalance::new(7, 10);
    balance.remove(NonZeroU32::new(3).unwrap()).unwrap();
    assert_eq!(
        balance.remove(NonZeroU32::new(5).unwrap()).unwrap_err(),
        CountArithmeticError::Underflow {
            original: 7,
            added: 10,
            removed: 8,
        }
    );
    assert!(!balance.is_zero());
    assert_eq!(balance.remaining().unwrap(), 14);
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn scalar_count_balances_allocate_no_storage() {
    let (control, positive) = allocation::measure("count_balance_allocating_control", || {
        std::hint::black_box(Vec::<u8>::with_capacity(std::hint::black_box(64)))
    });
    assert!(control.capacity() >= 64);
    assert!(positive.blocks > 0 && positive.bytes >= 64, "{positive}");
    let (empty, measured) = allocation::measure("empty_count_balance", || {
        let balance = CountBalance::new(0, 0);
        (balance.is_zero(), balance.remaining())
    });
    assert_eq!(empty, (true, Ok(0)));
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
    let (net, measured) = allocation::measure("nonempty_count_balance", || {
        let mut balance = CountBalance::new(u32::MAX, 1);
        balance.remove(NonZeroU32::MIN).unwrap();
        balance.remaining()
    });
    assert_eq!(net.unwrap(), u32::MAX);
    assert_eq!(measured.blocks, 0, "{measured}");
    assert_eq!(measured.bytes, 0, "{measured}");
}
