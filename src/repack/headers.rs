//! Bounded retention of parsed discovery headers for consumption during planning.

use core::mem;

use crate::object_graph::ObjectId;
use crate::reader::ParsedHeader;

/// Parsed object headers indexed by normalized source identity.
///
/// Retained storage includes entry-vector capacity and the allocations of each parsed header.
/// Headers are removed on lookup. Entry storage is released after the last header is removed.
pub(super) struct HeaderCache {
    entries: Vec<Entry>,
    /// The retained bytes, including vacant entry capacity.
    bytes: usize,
    /// The number of entries whose header is present.
    remaining: usize,
    budget: usize,
}

impl HeaderCache {
    /// Creates an empty cache with a retained-byte limit of `budget`.
    pub(super) fn new(budget: usize) -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            remaining: 0,
            budget,
        }
    }

    /// Retains `header` if its allocations and any entry capacity fit the budget.
    ///
    /// The caller sorts entries with [`Self::sort`] after insertion and before lookup.
    pub(super) fn insert(&mut self, id: ObjectId, header: ParsedHeader) {
        let Self {
            entries,
            bytes,
            remaining,
            budget,
        } = self;
        let payload = header.allocation_bytes();
        let Some(spare) = budget
            .checked_sub(*bytes)
            .and_then(|spare| spare.checked_sub(payload))
        else {
            return;
        };
        if entries.len() == entries.capacity() {
            let growth = entries
                .capacity()
                .max(INITIAL_ENTRIES)
                .min(spare / mem::size_of::<Entry>());
            if growth == 0 {
                return;
            }
            let previous = entries.capacity();
            entries.reserve_exact(growth);
            *bytes += (entries.capacity() - previous) * mem::size_of::<Entry>();
        }
        *bytes += payload;
        *remaining += 1;
        entries.push(Entry {
            id,
            header: Some(header),
        });
    }

    /// Sorts entries by normalized source identity for lookup.
    pub(super) fn sort(&mut self) {
        self.entries.sort_unstable_by_key(|entry| entry.id);
    }

    /// Removes headers whose identities fail `keep`, preserving entry order.
    pub(super) fn retain(&mut self, mut keep: impl FnMut(ObjectId) -> bool) {
        let Self {
            entries,
            bytes,
            remaining,
            budget: _,
        } = self;
        entries.retain(|Entry { id, header }| {
            if keep(*id) {
                return true;
            }
            if let Some(header) = header {
                *bytes -= header.allocation_bytes();
                *remaining -= 1;
            }
            false
        });
        self.release_empty();
    }

    /// Removes the header at `id` from the sorted entries, if present.
    pub(super) fn take(&mut self, id: ObjectId) -> Option<ParsedHeader> {
        let Self {
            entries,
            bytes,
            remaining,
            budget: _,
        } = self;
        let index = entries.binary_search_by_key(&id, |entry| entry.id).ok()?;
        let Entry { id: _, header } = entries.get_mut(index)?;
        let header = header.take()?;
        *bytes -= header.allocation_bytes();
        *remaining -= 1;
        self.release_empty();
        Some(header)
    }

    /// Releases all entries and their storage while preserving the budget.
    pub(super) fn clear(&mut self) {
        *self = Self::new(self.budget);
    }

    fn release_empty(&mut self) {
        let Self {
            entries,
            bytes,
            remaining,
            budget: _,
        } = self;
        if *remaining == 0 {
            *entries = Vec::new();
            *bytes = 0;
        }
    }
}

/// A source identity and its cached header, or a vacant entry after consumption.
struct Entry {
    id: ObjectId,
    header: Option<ParsedHeader>,
}

const INITIAL_ENTRIES: usize = 16;

#[cfg(all(test, any(target_arch = "x86_64", target_arch = "aarch64")))]
mod tests {
    use core::mem;

    use hdf5_pure_core::__private::PhysicalAddress;
    use rstest::rstest;
    use test_util::allocation;

    use super::Entry;
    use super::HeaderCache;
    use crate::AttrValue;
    use crate::File;
    use crate::FileBuilder;
    use crate::object_graph::ObjectId;

    #[rstest]
    #[case::disabled(1, 0, 0, 0, "header_cache_disabled", "header_cache_disabled_control")]
    #[case::below(1, 1, 1, 0, "header_cache_below", "header_cache_below_control")]
    #[case::exact(1, 1, 0, 1, "header_cache_exact", "header_cache_exact_control")]
    #[case::growth(64, 32, 0, 24, "header_cache_growth", "header_cache_growth_control")]
    fn retained_allocations_fit_the_header_cache_budget(
        #[case] count: usize,
        #[case] entries: usize,
        #[case] shortfall: usize,
        #[case] retained: usize,
        #[case] region: &'static str,
        #[case] control_region: &'static str,
    ) {
        let mut builder = FileBuilder::new();
        for index in 0..count {
            builder
                .create_dataset(&format!("values{index:04}"))
                .with_i32_data(&[3, 7])
                .set_attr("index", AttrValue::I32(i32::try_from(index).unwrap()));
        }
        let file = File::from_bytes(builder.finish().unwrap()).unwrap();
        let addresses: Vec<_> = (0..count)
            .map(|index| {
                PhysicalAddress::new(
                    file.dataset(&format!("values{index:04}"))
                        .unwrap()
                        .header_address()
                        .unwrap(),
                )
            })
            .collect();
        let (_, _, header) = file.hard_links_at(addresses[0]).unwrap();
        let payload = header.allocation_bytes();
        drop(header);
        let budget = if entries == 0 {
            0
        } else {
            payload * retained.max(1) + entries * mem::size_of::<Entry>() - shortfall
        };
        let (control, positive) =
            allocation::measure(control_region, || std::hint::black_box(vec![0_u8; 1024]));
        assert!(positive.blocks > 0, "{positive}");
        assert!(positive.peak_bytes >= 1024, "{positive}");
        drop(control);
        let (mut cache, measured) = allocation::measure(region, || {
            let mut cache = HeaderCache::new(budget);
            for &address in &addresses {
                let (_, _, header) = file.hard_links_at(address).unwrap();
                assert_eq!(header.allocation_bytes(), payload);
                cache.insert(ObjectId::new(address), header);
            }
            cache.sort();
            cache
        });
        assert!(measured.peak_bytes >= payload as u64, "{measured}");
        assert_eq!(measured.live_bytes, cache.bytes as u64, "{measured}");
        assert!(cache.bytes <= budget, "{measured}");
        assert_eq!(cache.remaining, retained);
        for (index, address) in addresses.into_iter().enumerate() {
            let header = cache.take(ObjectId::new(address));
            assert_eq!(header.is_some(), index < retained);
            let attributes = file.attr_messages_at(address, header).unwrap();
            assert_eq!(attributes.len(), 1);
            assert_eq!(attributes[0].name, "index");
            assert_eq!(
                attributes[0].raw_data,
                i32::try_from(index).unwrap().to_le_bytes()
            );
        }
        assert_eq!(cache.bytes, 0);
        assert_eq!(cache.entries.capacity(), 0);
        assert_eq!(cache.remaining, 0);
    }
}
