use crate::{
    ast::{Ci, Cn},
    parsers::math_expression_tree::MathExpressionTree,
};
use derive_new::new;
use std::fmt;

/// Content MathML expressions, with some interpretation.
/// https://www.w3.org/TR/MathML3/appendixa.html#parsing_ContExp
#[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Clone, Hash, new)]
pub enum ContExp {
    Ci(Ci),
    Cn(Cn),

    /// We deviate from the W3 spec with the Apply variant for convenience.
    Apply(Operator, MathExpressionTree),
}

/// Derivative operator, in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct Derivative {
    pub order: u8,
    pub var_index: u8,
}

/// Domain
/// https://www.w3.org/TR/MathML3/appendixa.html#parsing_DomainQ
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub enum Domain {
    DomainOfApplication(Box<ContExp>),
    Condition(MathExpressionTree),
    Limits { lower: ContExp, upper: ContExp },
}

/// Summation operator, structured for easy conversion to content MathML
/// (https://www.w3.org/TR/MathML3/chapter4.html#contm.sum).
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct Sum {
    pub bound_variables: Vec<Ci>,
    pub domain: Domain,
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub enum Operator {
    Add,
    Multiply,
    Equals,
    Divide,
    Subtract,
    Sqrt,
    Lparen,
    Rparen,
    Compose,
    Factorial,
    Derivative(Derivative),
    Sum(Sum),

    /// Set construction operator
    /// https://www.w3.org/TR/MathML3/chapter4.html#contm.sets
    Set,

    /// Set inclusion operator
    /// https://www.w3.org/TR/MathML3/chapter4.html#contm.in
    In,
    /// Catchall for operators we haven't explicitly defined as enum variants yet.
    Other(String),
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Add => write!(f, "+"),
            Operator::Multiply => write!(f, "*"),
            Operator::Equals => write!(f, "="),
            Operator::Divide => write!(f, "/"),
            Operator::Subtract => write!(f, "-"),
            Operator::Sqrt => write!(f, "√"),
            Operator::Lparen => write!(f, "("),
            Operator::Rparen => write!(f, ")"),
            Operator::Compose => write!(f, "."),
            Operator::Factorial => write!(f, "!"),
            Operator::In => write!(f, "∈"),
            Operator::Derivative(Derivative { order, var_index }) => {
                write!(f, "D({order}, {var_index})")
            }
            Operator::Sum(sum) => {
                write!(f, "{sum:?}")
            }
            Operator::Other(op) => write!(f, "{op:?}"),
        }
    }
}
