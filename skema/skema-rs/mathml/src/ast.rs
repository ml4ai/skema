use derive_new::new;
use schemars::JsonSchema;
use std::fmt;
use utoipa::ToSchema;
pub mod operator;
use operator::Operator;
use serde::{Deserialize, Serialize};
//use crate::ast::MathExpression::SummationOp;

/// Represents identifiers such as variables, function names, constants.
/// E.g. x , sin, n
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct Mi(pub String);

/// Represents groups of any subexpressions
/// E.g. For a given expression: 2x+4 -->  2 and x are grouped together
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct Mrow(pub Vec<MathExpression>);

#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub enum Type {
    Integer,
    Rational,
    Real,
    Complex,
    ComplexPolar,
    ComplexCartesian,
    Constant,
    Function,
    Vector,
    List,
    Set,
    Matrix,
}

/// Represents content identifiers such that variables can be have type attribute
/// to specify what it represents. E.g. function, vector, real,...
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct Ci {
    pub r#type: Option<Type>,
    pub content: Box<MathExpression>,
    pub func_of: Option<Vec<Ci>>,
    pub notation: Option<VectorNotation>,
}

/// Vector notation call for Ci being bold or an arrow
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub enum VectorNotation {
    Bold,
    Arrow,
}

/// Represents the differentiation operator `diff` for functions or expresions `func`
/// E.g. df/dx
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct Differential {
    pub diff: Box<MathExpression>,
    pub func: Box<MathExpression>,
}

/// Represents sum operator with terms that are summed over
/// E.g. sum_{i=1}^{K} f(x_i)
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct SummationMath {
    pub op: Box<MathExpression>,
    pub func: Box<MathExpression>,
}

/// Represents expoenential operator with terms that are in the exponential
/// E.g. exp( x(y-z) )
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct ExpMath {
    pub op: Box<MathExpression>,
    pub func: Box<MathExpression>,
}

/// Integral operation represents integral over math expressions
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct Integral {
    pub op: Box<MathExpression>,
    pub integrand: Box<MathExpression>,
    pub integration_variable: Box<MathExpression>,
}

/// Represents Hat operator and the contents the hat operator is being applied to.
/// E.g. f \hat{x}, where `op` will be \hat{x}, `comp` with be the contents
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct HatComp {
    pub op: Box<MathExpression>,
    pub comp: Box<MathExpression>,
}

/// Laplacian operator of vector calculus which takes in argument `comp`.
/// E.g. ∇^2 f
#[derive(
    Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize, JsonSchema,
)]
pub struct LaplacianComp {
    pub op: Box<MathExpression>,
    pub comp: Ci,
}

/// Handles ↓  such that {comp}↓_{sub}^{sup} can parse
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
pub struct DownArrow {
    pub sub: Option<Box<MathExpression>>,
    pub sup: Option<Box<MathExpression>>,
    pub comp: Box<MathExpression>,
}

/// The MathExpression enum is not faithful to the corresponding element type in MathML 3
/// (https://www.w3.org/TR/MathML3/appendixa.html#parsing_MathExpression)
#[derive(
    Debug,
    PartialOrd,
    Ord,
    PartialEq,
    Eq,
    Clone,
    Hash,
    Default,
    new,
    Deserialize,
    Serialize,
    ToSchema,
    JsonSchema,
)]
pub enum MathExpression {
    Mi(Mi),
    Mo(Operator),
    /// Represents numeric literals such as integers, decimals
    Mn(String),
    /// Represents square root elements
    Msqrt(Box<MathExpression>),
    Mrow(Mrow),
    /// Represents the fraction of expressions where first argument is numerator content and second argument is denominator content
    Mfrac(Box<MathExpression>, Box<MathExpression>),
    /// Represents expressions where first argument is the base and second argument is the superscript
    Msup(Box<MathExpression>, Box<MathExpression>),
    /// Represents expressions where first argument is the base and second argument is the subscript
    Msub(Box<MathExpression>, Box<MathExpression>),
    /// Represents expressions where first argument is the base and second argument is the undersubscript
    Munder(Box<MathExpression>, Box<MathExpression>),
    /// Represents expressions where first argument is the base and second argument is the oversubscript
    Mover(Box<MathExpression>, Box<MathExpression>),
    /// Represents expressions where first argument is the base, second argument is the subscript, third argument is the superscript
    Msubsup(
        Box<MathExpression>,
        Box<MathExpression>,
        Box<MathExpression>,
    ),
    /// Represents arbitrary text
    Mtext(String),
    /// Represents how elements can make changes to their style. E.g. displaystyle
    Mstyle(Vec<MathExpression>),
    /// Represents empty element with blanck space
    Mspace(String),
    /// Handles <mo .../>
    MoLine(String),
    Ci(Ci),
    Differential(Differential),
    SummationMath(SummationMath),
    AbsoluteSup(Box<MathExpression>, Box<MathExpression>),
    Absolute(Box<MathExpression>, Box<MathExpression>),
    HatComp(HatComp),
    Integral(Integral),
    LaplacianComp(LaplacianComp),
    /// Represents closed surface integral over contents. E.g. \\oiint_S ∇ \cdot T dS
    SurfaceIntegral(Box<MathExpression>),
    /// ↓ as an operator e.g. I↓ indicates downward diffuse radiative fluxes per unit indcident flux
    DownArrow(DownArrow),
    Minimize(Box<MathExpression>, Vec<MathExpression>),
    ExpMath(ExpMath),
    #[default]
    None,
}

impl fmt::Display for MathExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathExpression::Mi(Mi(identifier)) => write!(f, "{}", identifier),
            MathExpression::Ci(Ci {
                r#type: _,
                content,
                func_of: _,
                notation: _,
            }) => write!(f, "{}", content),
            MathExpression::Mn(number) => write!(f, "{}", number),
            MathExpression::Msup(base, superscript) => {
                write!(f, "{base}^{{{superscript}}}")
            }
            MathExpression::Msub(base, subscript) => {
                write!(f, "{base}_{{{subscript}}}")
            }
            MathExpression::Msubsup(base, subscript, superscript) => {
                write!(f, "{base}_{{{subscript}}}^{{{superscript}}}")
            }
            MathExpression::Mo(op) => {
                write!(f, "{}", op)
            }
            MathExpression::Mrow(Mrow(elements)) => {
                for e in elements {
                    write!(f, "{}", e)?;
                }
                Ok(())
            }
            MathExpression::Differential(Differential { diff, func }) => {
                write!(f, "{diff}")?;
                write!(f, "{func}")
            }
            MathExpression::AbsoluteSup(base, superscript) => {
                write!(f, "{:?}", base)?;
                write!(f, "{superscript:?}")
            }
            MathExpression::Mtext(text) => write!(f, "{}", text),
            MathExpression::SummationMath(SummationMath { op, func }) => {
                write!(f, "{op}")?;
                write!(f, "{func}")
            }
            MathExpression::ExpMath(ExpMath { op, func }) => {
                write!(f, "{op}")?;
                write!(f, "{func}")
            }
            MathExpression::HatComp(HatComp { op, comp }) => {
                write!(f, "{op}")?;
                write!(f, "{comp}")
            }
            MathExpression::Integral(Integral {
                op,
                integrand,
                integration_variable,
            }) => {
                write!(f, "{op}")?;
                write!(f, "{integrand}")?;
                write!(f, "{integration_variable}")
            }
            MathExpression::LaplacianComp(LaplacianComp { op, comp }) => {
                write!(f, "{op}(")?;
                write!(f, "{comp})")
            }
            MathExpression::SurfaceIntegral(row) => {
                write!(f, "{row})")
            }
            MathExpression::DownArrow(DownArrow { sub, sup, comp }) => match (sub, sup) {
                (Some(low), Some(up)) => write!(f, "{comp}↓_{{{low}}}^{{{up}}}"),
                (Some(low), None) => write!(f, "{comp}↓_{{{low}}}"),
                (None, Some(up)) => write!(f, "{comp}↓^{{{up}}}"),
                (None, None) => write!(f, "{comp}↓"),
            },
            MathExpression::Minimize(op, row) => {
                for e in row {
                    write!(f, "{}", e)?;
                }
                Ok(())
            }
            expression => write!(f, "{expression:?}"),
        }
    }
}

impl fmt::Display for Ci {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

/// The Math struct represents the corresponding element type in MathML 3
/// (https://www.w3.org/TR/MathML3/appendixa.html#parsing_math)
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Math {
    pub content: Vec<MathExpression>,
}
