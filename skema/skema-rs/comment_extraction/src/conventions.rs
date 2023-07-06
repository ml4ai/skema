use clap::ValueEnum;
pub mod dssat;

/// Some codebases (e.g., the codebase for the DSSAT crop modeling
/// system) follow a particular commenting convention. In such cases, we may want to handle the
/// comment extraction differently in order to facilitate alignment with the literature. This enum
/// contains the different conventions we handle.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, ValueEnum)]
pub enum Convention {
    DSSAT,
}
