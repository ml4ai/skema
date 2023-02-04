pub mod recognizers;
use crate::ast::MathExpression;

#[derive(Debug, PartialEq, Clone)]
pub struct Flux {
    rate: String,
    vars: Vec<MathExpression>,
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Specie(pub MathExpression);

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Rate(pub MathExpression);

#[derive(Debug, PartialEq, Clone)]
pub struct Transition {
    name: String,
    rate: Rate,
}

#[derive(Debug, PartialEq, Clone)]
pub struct PetriNet {
    species: Vec<Specie>,
    transitions: Vec<Transition>,
    inputs: (Specie, Transition, u32),
    outputs: (Specie, Transition, u32),
}

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
