//! Resolved ZFP settings retained between chunk operations.

#![cfg(feature = "zfp")]

use alloc::vec::Vec;

use super::{ChunkContext, FilterScratch};
use crate::error::Error;
use crate::zfp::ResolvedZfpSettings;

impl FilterScratch {
    /// Serializes ZFP client data and retains the settings at the pipeline's `index`.
    pub fn zfp_client_data(&mut self, settings: ResolvedZfpSettings, index: usize) -> Vec<u32> {
        let client_data = settings.client_data();
        self.store_zfp(settings, &client_data, index);
        client_data
    }

    pub(super) fn zfp_settings(
        &mut self,
        ctx: ChunkContext<'_>,
        client_data: &[u32],
        index: usize,
    ) -> Result<ResolvedZfpSettings, Error> {
        if let Some(prepared) = self.zfp.iter().find(|prepared| prepared.index == index) {
            if prepared.client_data == client_data
                && prepared
                    .settings
                    .matches_context(ctx.chunk_dims, ctx.element_type)
            {
                return Ok(prepared.settings);
            }
        }
        let settings =
            ResolvedZfpSettings::from_filter(client_data, ctx.chunk_dims, ctx.element_type)?;
        self.store_zfp(settings, client_data, index);
        Ok(settings)
    }

    fn store_zfp(&mut self, settings: ResolvedZfpSettings, client_data: &[u32], index: usize) {
        if let Some(prepared) = self.zfp.iter_mut().find(|prepared| prepared.index == index) {
            prepared.client_data.clear();
            prepared.client_data.extend_from_slice(client_data);
            prepared.settings = settings;
        } else {
            self.zfp.push(PreparedZfpFilter {
                index,
                client_data: client_data.to_vec(),
                settings,
            });
        }
        #[cfg(test)]
        {
            self.zfp_preparations += 1;
        }
    }
}

pub(super) struct PreparedZfpFilter {
    index: usize,
    client_data: Vec<u32>,
    settings: ResolvedZfpSettings,
}
