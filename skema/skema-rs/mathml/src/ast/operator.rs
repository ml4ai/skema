use derive_new::new;
use std::fmt;

/// Derivative operator, in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct Derivative {
    pub order: u8,
    pub var_index: u8,
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
            Operator::Factorial => write!(f, "!"),
            Operator::Derivative(Derivative { order, var_index }) => {
                write!(f, "D({order}, {var_index})")
            }
            Operator::Other(op) => write!(f, "{op}"),
        }
    }
}
