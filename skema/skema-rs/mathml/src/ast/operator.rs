use crate::ast::Ci;
use derive_new::new;
use std::fmt;

/// Derivative operator, in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct Derivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
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
    Exp,
    Power,
    Comma,
    Grad,
    Dot,
    Derivative(Derivative),
    Sin,
    Cos,
    Tan,
    Sec,
    Csc,
    Cot,
    Arcsin,
    Arccos,
    Arctan,
    Arcsec,
    Arccsc,
    Arccot,
    Mean,
    // Catchall for operators we haven't explicitly defined as enum variants yet.
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
            Operator::Comma => write!(f, ","),
            Operator::Factorial => write!(f, "!"),
            Operator::Derivative(Derivative {
                order,
                var_index,
                bound_var,
            }) => {
                write!(f, "D({order}, {var_index}, {bound_var})")
            }
            Operator::Exp => write!(f, "exp"),
            Operator::Power => write!(f, "^"),
            Operator::Other(op) => write!(f, "{op}"),
            Operator::Sin => write!(f, "Sin"),
            Operator::Cos => write!(f, "Cos"),
            Operator::Tan => write!(f, "Tan"),
            Operator::Sec => write!(f, "Sec"),
            Operator::Csc => write!(f, "Csc"),
            Operator::Cot => write!(f, "Cot"),
            Operator::Arcsin => write!(f, "Arcsin"),
            Operator::Arccos => write!(f, "Arccos"),
            Operator::Arctan => write!(f, "Arctan"),
            Operator::Arcsec => write!(f, "Arcsec"),
            Operator::Arccsc => write!(f, "Arccsc"),
            Operator::Arccot => write!(f, "Arccot"),
            Operator::Mean => write!(f, "Mean"),
            Operator::Grad => write!(f, "Grad"),
            Operator::Dot => write!(f, "Dot"),
        }
    }
}
