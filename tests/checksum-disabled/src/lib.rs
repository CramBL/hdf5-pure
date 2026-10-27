#[cfg(test)]
mod tests {
    use hdf5_pure::Error;
    use hdf5_pure::File;
    use hdf5_pure::FileAccessProperties;
    use hdf5_pure::FileBuilder;
    use hdf5_pure::FileSpaceSettings;
    use hdf5_pure::FileSpaceStrategy;
    use hdf5_pure::FormatError;
    use hdf5_pure::MemoryStrategy;
    use hdf5_pure::PersistedFreeSpacePolicy;
    use hdf5_pure_format::__private as format;
    use rstest::rstest;

    #[rstest]
    fn strict_open_checks_persisted_checksums_without_the_checksum_feature(
        #[values(false, true)] header: bool,
        #[values(
            MemoryStrategy::Bounded,
            MemoryStrategy::Mirrored,
            MemoryStrategy::Auto
        )]
        memory: MemoryStrategy,
    ) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checksums.h5");
        let mut builder = FileBuilder::new();
        builder.with_file_space_settings(
            FileSpaceSettings::new(FileSpaceStrategy::FsmAggr).with_persistence(true),
        );
        builder.create_dataset("a").with_i32_data(&[1; 100]);
        builder.create_dataset("b").with_i32_data(&[2; 400]);
        builder.create_dataset("c").with_i32_data(&[3; 100]);
        builder.write(&path).unwrap();
        let file = File::open_rw(&path).unwrap();
        file.root().delete("b").unwrap();
        file.commit().unwrap();
        file.close().unwrap();
        let file = File::open(&path).unwrap();
        let report = file.persisted_free_sections().unwrap().unwrap();
        let manager = &report.managers()[0];
        let slot = manager.slot();
        let address = manager.address();
        drop(file);
        let mut bytes = std::fs::read(&path).unwrap();
        let widths = format::FormatWidths::from_sizes(8, 8).unwrap();
        let at = usize::try_from(address.get()).unwrap();
        let parsed = format::FreeSpaceManagerHeader::parse(widths, &bytes[at..]).unwrap();
        let list = parsed.section_list_addr();
        let checksum_at = if header {
            at + usize::try_from(format::free_space_manager_header_len(widths)).unwrap() - 4
        } else {
            usize::try_from(list.get() + parsed.section_list_used()).unwrap() - 4
        };
        let checksum = u32::from_le_bytes(bytes[checksum_at..checksum_at + 4].try_into().unwrap());
        bytes[checksum_at] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        let opened = File::open_rw_with_options(
            &path,
            FileAccessProperties::new()
                .with_memory_strategy(memory)
                .with_persisted_free_space_policy(PersistedFreeSpacePolicy::Strict),
        );
        let error = opened.unwrap_err();
        let Error::PersistedFreeSpace {
            manager_slot,
            header_address,
            section_list_address,
            section,
            source,
        } = error
        else {
            panic!("{error:?}")
        };
        assert_eq!(
            (manager_slot, header_address, section_list_address, section),
            (slot, address, (!header).then_some(list), None)
        );
        let Error::Format(FormatError::ChecksumMismatch { expected, computed }) = *source else {
            panic!("{source:?}")
        };
        assert_eq!((expected, computed), (checksum ^ 1, checksum));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        let widths = format::FormatWidths::from_sizes(8, 8).unwrap();
        if header {
            let parsed = format::FreeSpaceManagerHeader::parse(widths, &bytes[at..]).unwrap();
            assert_eq!(parsed.section_list_addr(), list);
        } else {
            let sections = format::parse_section_info(
                widths,
                &bytes[usize::try_from(list.get()).unwrap()..],
                address,
                &parsed,
            )
            .unwrap();
            assert_eq!(sections.len(), manager.free_sections().section_count());
        }
    }
}
