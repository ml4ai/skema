use crate::ast::Ci;
use crate::ast::MathExpression;
use derive_new::new;
use std::fmt;
use serde::{Deserialize, Serialize};

/// Derivative operator, in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Derivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
}

/// Partial derivative operator
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct PartialDerivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
}

/// Summation operator with under and over components
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct SumUnderOver {
    pub op: Box<MathExpression>,
    pub under: Box<MathExpression>,
    pub over: Box<MathExpression>,
}

/// Hat operation
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct HatOp {
    pub comp: Box<MathExpression>,
}

/// Handles grad operations with subscript. E.g. ∇_{x}
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct GradSub {
    pub sub: Box<MathExpression>,
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
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
    GradSub(GradSub),
    Dot,
    Period,
    Div,
    Abs,
    Derivative(Derivative),
    PartialDerivative(PartialDerivative),
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
    Sum,
    SumUnderOver(SumUnderOver),
    Cross,
    Hat,
    HatOp(HatOp),
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
                var_index: _,
                bound_var,
            }) => {
                write!(f, "D({order}, {bound_var})")
            }
            Operator::PartialDerivative(PartialDerivative {
                order,
                var_index: _,
                bound_var,
            }) => {
                write!(f, "PD({order}, {bound_var})")
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
            Operator::GradSub(GradSub {sub}) =>{
                write!(f, "Grad_{sub})")
            }
            Operator::Dot => write!(f, "⋅"),
            Operator::Period => write!(f, ""),
            Operator::Div => write!(f, "Div"),
            Operator::Abs => write!(f, "Abs"),
            Operator::Sum => write!(f, "∑"),
            Operator::SumUnderOver(SumUnderOver { op, under, over }) => {
                write!(f, "{op}_{{{under}}}^{{{over}}}")
            }
            Operator::Cross => write!(f, "×"),
            Operator::Hat => write!(f, "Hat"),
            Operator::HatOp(HatOp { comp }) => write!(f, "Hat({comp})"),
        }
    }
}
