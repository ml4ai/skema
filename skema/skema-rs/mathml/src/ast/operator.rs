use crate::ast::Ci;
use crate::ast::MathExpression;
use derive_new::new;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;
use utoipa::ToSchema;

/// Derivative operator, in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct Derivative {
    pub order: u8,
    pub var_index: u8,
    pub bound_var: Ci,
    pub notation: DerivativeNotation,
}

/// All the stuff that was in the LaTeX equation that we need to reproduce the original LaTeX equation
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub enum DerivativeNotation {
    /// e.g. df/dx
    LeibnizTotal,
    /// e.g. ∂f/∂x
    LeibnizPartialStandard,
    /// e.g ∂_x
    LeibnizPartialCompact,
    /// dot notation, e.g. \dot{f}
    Newton,
    /// Df/Dx
    DNotation,
    /// prime notation, e.g. f'
    Lagrange,
}

#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct Logarithm {
    pub notation: LogarithmNotation,
    //pub is_natural_log: bool,
    //pub base: Option<Box<MathExpression>>,
}

/// Varying logarithm notation
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub enum LogarithmNotation {
    /// e.g. ln
    Ln,
    /// e.g log
    Log,
    /// e.g. log with base
    LogBase(Box<MathExpression>),
}

/// Summation operator has the option of having lower bound and upper bound components
#[derive(
    Debug,
    Ord,
    PartialOrd,
    PartialEq,
    Eq,
    Clone,
    Hash,
    new,
    Deserialize,
    Serialize,
    ToSchema,
    JsonSchema,
)]
pub struct Summation {
    pub lower_bound: Option<Box<MathExpression>>,
    pub upper_bound: Option<Box<MathExpression>>,
}

/// Hat operation obtains the hat operation with the operation component: e.g. \hat{x}
#[derive(
    Debug,
    Ord,
    PartialOrd,
    PartialEq,
    Eq,
    Clone,
    Hash,
    new,
    Deserialize,
    Serialize,
    ToSchema,
    JsonSchema,
)]
pub struct Hat {
    pub comp: Box<MathExpression>,
}

/// Gradient operation has the option of handling grad operations with subscript. E.g. ∇_{x}
#[derive(
    Debug,
    Ord,
    PartialOrd,
    PartialEq,
    Eq,
    Clone,
    Hash,
    new,
    Deserialize,
    Serialize,
    ToSchema,
    JsonSchema,
)]
pub struct Gradient {
    pub subscript: Option<Box<MathExpression>>,
}

/// Integral can be definite or indefinite with `integration_variable`
/// as it has the option of having `lower_limit`, `upper_limit`
#[derive(
    Debug,
    Ord,
    PartialOrd,
    PartialEq,
    Eq,
    Clone,
    Hash,
    new,
    Deserialize,
    Serialize,
    ToSchema,
    JsonSchema,
)]
pub struct Int {
    pub lower_limit: Option<Box<MathExpression>>,
    pub upper_limit: Option<Box<MathExpression>>,
    pub integration_variable: Box<MathExpression>,
}

#[derive(
    Debug,
    Ord,
    PartialOrd,
    PartialEq,
    Eq,
    Clone,
    Hash,
    new,
    Deserialize,
    Serialize,
    ToSchema,
    JsonSchema,
)]
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
    /// Square root operator
    Sqrt,
    /// Left parenthesis; only for internal use (Pratt parsing)
    Lparen,
    /// Right parenthesis; only for internal use (Pratt parsing)
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
    /// Divergence operator
    Div,
    ///Absolute value
    Abs,
    /// Includes derivative operator with varying derivative notation
    Derivative(Derivative),
    /// Sine function
    Sin,
    /// Cosine function
    Cos,
    /// Tangent function
    Tan,
    /// Secant function
    Sec,
    /// Cosecant function
    Csc,
    /// Cotangent function
    Cot,
    /// Arcsine function
    Arcsin,
    /// Arccosine function
    Arccos,
    /// Arctangent function
    Arctan,
    /// Arcsecant function
    Arcsec,
    /// Arccosecant function
    Arccsc,
    /// Arccotangent function
    Arccot,
    /// Mean operator
    Mean,
    /// Summation operator
    Summation(Summation),
    /// Cross product operator
    Cross,
    /// Hat operator with component, e.g. \hat{x}
    Hat(Hat),
    /// Integrals
    Int(Int),
    /// Laplacian operator
    Laplacian,
    /// Closed surface integral operator --need to include explit dS integration variable when translating to latex
    SurfaceInt,
    /// only for internal use
    Comma,
    /// Logarithm operator
    Logarithm(Logarithm),
    /// Minimum operator
    Min,
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
            Operator::Derivative(Derivative {
                order,
                var_index: _,
                bound_var,
                notation,
            }) => match notation {
                DerivativeNotation::LeibnizTotal => write!(f, "D({order}, {bound_var})"),
                DerivativeNotation::LeibnizPartialStandard => write!(f, "PD({order}, {bound_var})"),
                DerivativeNotation::LeibnizPartialCompact => write!(f, "∂_{{{bound_var}}})"),
                DerivativeNotation::Newton => write!(f, "D({order}, {bound_var})"),
                DerivativeNotation::Lagrange => write!(f, "D({order}, {bound_var})"),
                DerivativeNotation::DNotation => write!(f, "D({order}, {bound_var})"),
            },
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
            Operator::Gradient(Gradient { subscript }) => match subscript {
                Some(sub) => write!(f, "Grad_{sub}"),
                None => write!(f, "Grad"),
            },
            Operator::Dot => write!(f, "⋅"),
            Operator::Div => write!(f, "Div"),
            Operator::Abs => write!(f, "Abs"),
            Operator::Summation(Summation {
                lower_bound,
                upper_bound,
            }) => match (lower_bound, upper_bound) {
                (Some(low), Some(up)) => write!(f, "Sum_{{{low}}}^{{{up}}}"),
                (Some(low), None) => write!(f, "Sum_{{{low}}}"),
                (None, Some(up)) => write!(f, "Sum^{{{up}}}"),
                (None, None) => write!(f, "Sum"),
            },
            Operator::Int(Int {
                lower_limit,
                upper_limit,
                integration_variable,
            }) => match (lower_limit, upper_limit) {
                (Some(low), Some(up)) => {
                    write!(f, "Int_{{{low}}}^{{{up}}}({integration_variable})")
                }
                (Some(low), None) => write!(f, "Int_{{{low}}}({integration_variable})"),
                (None, Some(up)) => write!(f, "Int^{{{up}}}(integration_variable)"),
                (None, None) => write!(f, "Int"),
            },
            Operator::Cross => write!(f, "×"),
            Operator::Hat(Hat { comp }) => write!(f, "Hat({comp})"),
            Operator::Laplacian => write!(f, "Laplacian"),
            Operator::SurfaceInt => {
                write!(f, "SurfaceInt")
            }
            Operator::Min => write!(f, "Min"),
            Operator::Comma => write!(f, ","),
            Operator::Logarithm(Logarithm { notation }) => match notation {
                LogarithmNotation::Ln => write!(f, "Ln"),
                LogarithmNotation::Log => write!(f, "Log"),
                LogarithmNotation::LogBase(x) => write!(f, "Log_{{{x}}}"),
            },
        }
    }
}
