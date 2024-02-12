use derive_new::new;
use std::fmt;

pub mod operator;
use operator::Operator;
use serde::{Deserialize, Serialize};
//use crate::ast::MathExpression::SummationOp;

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Mi(pub String);

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Mrow(pub Vec<MathExpression>);

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
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

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Ci {
    pub r#type: Option<Type>,
    pub content: Box<MathExpression>,
    pub func_of: Option<Vec<Ci>>,
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Differential {
    pub diff: Box<MathExpression>,
    pub func: Box<MathExpression>,
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct SummationMath {
    pub op: Box<MathExpression>,
    pub func: Box<MathExpression>,
}

/// Integral operation
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct Integral {
    pub op: Box<MathExpression>,
    pub integrand: Box<MathExpression>,
    pub integration_variable: Box<MathExpression>,
}

/// Integral operation
/*#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct SurfaceClosedIntegral {
    //pub op: Box<MathExpression>,
    pub row: Mrow,
}*/

/// Hat operation
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct HatComp {
    pub op: Box<MathExpression>,
    pub comp: Box<MathExpression>,
}

/// DownArrow
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct DownArrowComp {
    pub op: Box<MathExpression>,
    pub comp: Box<MathExpression>,
    pub func_of: Vec<Ci>,
}

/// Laplacian
#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new, Deserialize, Serialize)]
pub struct LaplacianComp {
    pub op: Box<MathExpression>,
    pub comp: Ci,
}

/// The MathExpression enum is not faithful to the corresponding element type in MathML 3
/// (https://www.w3.org/TR/MathML3/appendixa.html#parsing_MathExpression)
#[derive(
    Debug, PartialOrd, Ord, PartialEq, Eq, Clone, Hash, Default, new, Deserialize, Serialize,
)]
pub enum MathExpression {
    Mi(Mi),
    Mo(Operator),
    Mn(String),
    Msqrt(Box<MathExpression>),
    Mrow(Mrow),
    Mfrac(Box<MathExpression>, Box<MathExpression>),
    Msup(Box<MathExpression>, Box<MathExpression>),
    Msub(Box<MathExpression>, Box<MathExpression>),
    Munder(Box<MathExpression>, Box<MathExpression>),
    Mover(Box<MathExpression>, Box<MathExpression>),
    Msubsup(
        Box<MathExpression>,
        Box<MathExpression>,
        Box<MathExpression>,
    ),
    Mtext(String),
    Mstyle(Vec<MathExpression>),
    Mspace(String),
    MoLine(String),
    //GroupTuple(Vec<MathExpression>),
    Ci(Ci),
    Differential(Differential),
    SummationMath(SummationMath),
    AbsoluteSup(Box<MathExpression>, Box<MathExpression>),
    Absolute(Box<MathExpression>, Box<MathExpression>),
    HatComp(HatComp),
    DownArrowComp(DownArrowComp),
    Integral(Integral),
    LaplacianComp(LaplacianComp),
    SurfaceClosedIntegral(Box<MathExpression>),
    SurfaceClosedIntegralNoIntVar(Box<MathExpression>),
    //Differential(Box<MathExpression>, Box<MathExpression>),
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
            MathExpression::HatComp(HatComp { op, comp }) => {
                write!(f, "{op}")?;
                write!(f, "{comp}")
            }
            MathExpression::DownArrowComp(DownArrowComp {
                op,
                comp,
                func_of: _,
            }) => {
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
            MathExpression::SurfaceClosedIntegral(row) => {
                write!(f, "{row})")
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
