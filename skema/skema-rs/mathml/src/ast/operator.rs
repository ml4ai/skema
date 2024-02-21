use crate::ast::Ci;
use crate::ast::MathExpression;
use derive_new::new;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Total Derivative operator, e.g. dS/dt . in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Derivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
}

/// D Derivative operator, e.g. DS/Dt . in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct DDerivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
}

/// Partial derivative operator. e.g. ∂S/∂t
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct PartialDerivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
}

/// Summation operator has the option of having lowlimit and uplimit components
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Summation {
    pub lowlimit: Option<Box<MathExpression>>,
    pub uplimit: Option<Box<MathExpression>>,
}

/// Hat operation obtains the hat operation with the operation component: e.g. \hat{x}
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct HatOp {
    pub comp: Box<MathExpression>,
}

/// Gradient operation has the option of handling grad operations with subscript. E.g. ∇_{x}
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Gradient {
    pub subscript: Option<Box<MathExpression>>,
}

/// Integral can be definite or indefinite with `integration_variable`
/// as it has the option of having `lowlimit`, `uplimit`
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Int {
    pub lowlimit: Option<Box<MathExpression>>,
    pub uplimit: Option<Box<MathExpression>>,
    pub integration_variable: Box<MathExpression>,
}

/// Handles ↓ as an operator such that {comp}↓_{sub}^{sup} can parse
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct DownArrow {
    pub sub: Option<Box<MathExpression>>,
    pub sup: Option<Box<MathExpression>>,
    pub comp: Box<MathExpression>,
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub enum Operator {
    /// Addition operator
    Add,
    /// Multiplication operator
    Multiply,
    /// Equals operator
    Equals,
    /// Division operator
    Divide,
    /// Subtraction operator
    Subtract,
    /// Squre root operator
    Sqrt,
    /// Left parenthesis
    Lparen,
    /// Right parenthesis
    Rparen,
    /// Composition operator
    Compose,
    /// Factorial operator
    Factorial,
    /// Exponential operator
    Exp,
    /// Power (exponent) operator
    Power,
    /// Gradient operator
    Gradient(Gradient),
    /// Dot product operator
    Dot,
    Comma,
    Period,
    /// Divergence operator
    Div,
    ///Absolute operator
    Abs,
    /// Total derivative operator, e.g. d/dt
    Derivative(Derivative),
    /// Partial derivative operator, e.g. ∂/∂t
    PartialDerivative(PartialDerivative),
    /// Partial derivative operator, e.g. D/Dt
    DDerivative(DDerivative),
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
    /// Mean operator
    Mean,
    /// Summation operator
    Summation(Summation),
    /// Cross product operator
    Cross,
    /// Hat operator, e.g. \hat
    Hat,
    /// Hat operator with component, e.g. \hat{x}
    HatOp(HatOp),
    /// ↓ as an operator
    DownArrow(DownArrow),
    /// Integrals
    Int(Int),
    /// Laplacian operator
    Laplacian,
    /// Closed surface integral operator --need to include explit dS integration variable when translating to latex
    SurfaceClosedInt,
    /// Closed surface integral operator -- doesn't need to include explicit dS integration variable when translating to latex
    /// E.g. \\oiint_S ∇ \cdot dS
    SurfaceClosedIntNoIntVar,
    Vector,
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
            Operator::DDerivative(DDerivative {
                order,
                var_index: _,
                bound_var,
            }) => {
                write!(f, "DD({order}, {bound_var})")
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
            Operator::Vector => write!(f, "Vec"),
            Operator::Gradient(Gradient { subscript }) => match subscript {
                Some(sub) => write!(f, "Grad_{sub}"),
                None => write!(f, "Grad"),
            },
            Operator::Dot => write!(f, "⋅"),
            Operator::Period => write!(f, ""),
            Operator::Div => write!(f, "Div"),
            Operator::Abs => write!(f, "Abs"),
            Operator::Summation(Summation { lowlimit, uplimit }) => match (lowlimit, uplimit) {
                (Some(low), Some(up)) => write!(f, "Sum_{{{low}}}^{{{up}}}"),
                (Some(low), None) => write!(f, "Sum_{{{low}}}"),
                (None, Some(up)) => write!(f, "Sum^{{{up}}}"),
                (None, None) => write!(f, "Sum"),
            },
            Operator::Int(Int {
                lowlimit,
                uplimit,
                integration_variable,
            }) => match (lowlimit, uplimit) {
                (Some(low), Some(up)) => {
                    write!(f, "Int_{{{low}}}^{{{up}}}({integration_variable})")
                }
                (Some(low), None) => write!(f, "Int_{{{low}}}({integration_variable})"),
                (None, Some(up)) => write!(f, "Int^{{{up}}}(integration_variable)"),
                (None, None) => write!(f, "Int"),
            },
            Operator::DownArrow(DownArrow { sub, sup, comp }) => match (sub, sup) {
                (Some(low), Some(up)) => write!(f, "{comp}↓_{{{low}}}^{{{up}}}"),
                (Some(low), None) => write!(f, "{comp}↓_{{{low}}}"),
                (None, Some(up)) => write!(f, "{comp}↓^{{{up}}}"),
                (None, None) => write!(f, "{comp}↓"),
            },
            Operator::Cross => write!(f, "×"),
            Operator::Hat => write!(f, "Hat"),
            Operator::HatOp(HatOp { comp }) => write!(f, "Hat({comp})"),
            Operator::Laplacian => write!(f, "Laplacian"),
            Operator::SurfaceClosedInt => {
                write!(f, "SurfaceClosedInt")
            }
            Operator::SurfaceClosedIntNoIntVar => {
                write!(f, "SurfaceClosedInt")
            }
        }
    }
}
