use super::StagedEdits;

#[cfg(test)]
impl StagedEdits {
    /// Returns the operation counts for equality checks in editor tests.
    pub(in crate::edit) fn snapshot(&self) -> impl Eq + use<> {
        self.mark()
    }
}
