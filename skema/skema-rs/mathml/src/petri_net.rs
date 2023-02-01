pub mod recognizers;
use crate::ast::MathExpression;

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct Specie(pub MathExpression);

#[derive(Debug, PartialEq, Clone)]
pub struct Transition {
    name: String,
    rate: f32,
}

#[derive(Debug, PartialEq, Default, Clone)]
pub struct PetriNet {
    species: Vec<Specie>,
    transitions: Vec<Transition>,
}

/// Represents the Tangent var of an ODE.
/// This is perhaps not really needed, although it at least introduces a type.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Tangent(pub Specie);

#[derive(Debug, Eq, PartialEq, Clone, Hash, Default)]
pub enum Polarity {
    #[default]
    add,
    sub,
}
