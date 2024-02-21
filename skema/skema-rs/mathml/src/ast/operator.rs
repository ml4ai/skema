use crate::ast::Ci;
use crate::ast::MathExpression;
use derive_new::new;
use serde::{Deserialize, Serialize};
use std::fmt;
use utoipa::ToSchema;
use schemars::JsonSchema;

/// Total Derivative operator, in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema)]
pub struct Derivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
}

/// Partial derivative operator
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema)]
pub struct PartialDerivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
}

/// Summation operator with under and over components
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema)]
pub struct SumUnderOver {
    pub op: Box<MathExpression>,
    pub under: Box<MathExpression>,
    pub over: Box<MathExpression>,
}

/// Hat operation obtains the hat operation with the operation component: e.g. \hat{x}
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema)]
pub struct HatOp {
    pub comp: Box<MathExpression>,
}

/// Handles grad operations with subscript. E.g. ∇_{x}
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema)]
pub struct GradSub {
    pub sub: Box<MathExpression>,
}

/// Definite Integral with lowlimit, uplimit
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema)]
pub struct MsubsupInt {
    pub lowlimit: Box<MathExpression>,
    pub uplimit: Box<MathExpression>,
    pub integration_variable: Box<MathExpression>,
}

/// MsupDownArrow operation. E.g. Handles I^{↓} operations such that I is `comp` of DownArrow operation
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema)]
pub struct MsupDownArrow {
    pub comp: Box<MathExpression>,
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, ToSchema, JsonSchema)]
pub enum Operator {
    /// Addition operator
    Add,
    Multiply,
    Equals,
    Divide,
    Subtract,
    Sqrt,
    /// Left parenthesis
    Lparen,
    /// Right parenthesis
    Rparen,
    Compose,
    Factorial,
    /// Exponential operator
    Exp,
    Power,
    Comma,
    /// Gradient operator
    Grad,
    GradSub(GradSub),
    Dot,
    Period,
    /// Divergence operator
    Div,
    ///Absolute operator
    Abs,
    /// Total derivative operator, e.g. d/dt
    Derivative(Derivative),
    /// Partial derivative operator, e.g. ∂/∂t
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
    /// Summation operator
    Sum,
    /// Summation operator with lowlimit and uplimit
    SumUnderOver(SumUnderOver),
    Cross,
    /// Hat operator, e.g. \hat
    Hat,
    /// Hat operator with component, e.g. \hat{x}
    HatOp(HatOp),
    /// Summation operator with lowlimit and uplimit
    MsupDownArrow(MsupDownArrow),
    /// ↓ as an operator
    DownArrow,
    /// Definite Integral
    Int,
    /// Indefinite Integral with lowlimit and uplimit
    MsubsupInt(MsubsupInt),
    Laplacian,
    /// Closed surface integral operator --need to include explit dS integration variable when translating to latex
    SurfaceClosedInt,
    /// Closed surface integral operator -- doesn't need to include explicit dS integration variable when translating to latex
    /// E.g. \\oiint_S ∇ \cdot dS
    SurfaceClosedIntNoIntVar,
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
            Operator::GradSub(GradSub { sub }) => {
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
            Operator::MsupDownArrow(MsupDownArrow { comp }) => write!(f, "{comp}↓"),
            Operator::DownArrow => write!(f, "↓"),
            Operator::Int => write!(f, "Int"),
            Operator::MsubsupInt(MsubsupInt {
                lowlimit,
                uplimit,
                integration_variable,
            }) => {
                write!(
                    f,
                    "Int_{{{lowlimit}}}^{{{uplimit}}}({integration_variable})"
                )
            }
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
