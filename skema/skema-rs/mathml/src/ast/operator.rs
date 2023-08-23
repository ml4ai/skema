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
    Derivative(Derivative),
    // Catchall for operators we haven't explicitly defined as enum variants yet.
    Other(String),
    Trig(Trig),
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub enum Trig {
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
            Operator::Exp => write!(f, "Exp"),
            Operator::Power => write!(f, "Power"),
            Operator::Other(op) => write!(f, "{op}"),
            Operator::Trig(Trig::Sin) => write!(f, "Sin"),
            Operator::Trig(Trig::Cos) => write!(f, "Cos"),
            Operator::Trig(Trig::Tan) => write!(f, "Tan"),
            Operator::Trig(Trig::Sec) => write!(f, "Sec"),
            Operator::Trig(Trig::Csc) => write!(f, "Csc"),
            Operator::Trig(Trig::Cot) => write!(f, "Cot"),
            Operator::Trig(Trig::Arcsin) => write!(f, "Arcsin"),
            Operator::Trig(Trig::Arccos) => write!(f, "Arccos"),
            Operator::Trig(Trig::Arctan) => write!(f, "Arctan"),
            Operator::Trig(Trig::Arcsec) => write!(f, "Arcsec"),
            Operator::Trig(Trig::Arccsc) => write!(f, "Arccsc"),
            Operator::Trig(Trig::Arccot) => write!(f, "Arccot"),
        }
    }
}
