//! Structs for representing elements of Petri Nets.

pub mod recognizers;
use crate::ast::MathExpression;
use std::fmt;

/// Representation of a "named" variable
/// Here, 'variable' is intended to mean a symbolic name for a value.
/// Variables could be names of species (states) or rate (parameters).
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Clone)]
pub struct Var(pub MathExpression);

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Clone)]
pub struct Specie(pub MathExpression);

impl fmt::Display for Specie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Hash, Clone, Default)]
pub struct Rate(pub MathExpression);

impl fmt::Display for Rate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Hash, Default)]
pub enum Polarity {
    #[default]
    Positive,
    Negative,
}
