pub mod recognizers;
use crate::ast::MathExpression;

/// Representation of a "named" variable
/// Here, 'variable' is intended to mean a symbolic name for a value.
/// Variables could be names of species (states) or rate (parameters).
#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Var(pub MathExpression);

#[derive(Debug, PartialEq, Clone)]
pub struct Flux {
    rate: String,
    vars: Vec<MathExpression>,
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Specie(pub MathExpression);

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Rate(pub MathExpression);

/// Represents the Tangent var of an ODE.
/// This is perhaps not really needed, although it at least introduces a type.
#[derive(Debug, Eq, PartialEq, Clone, Hash)]
pub struct Tangent(pub Specie);

#[derive(Debug, Eq, PartialEq, Clone, Hash, Default)]
pub enum Polarity {
    #[default]
    add,
    sub,
}
