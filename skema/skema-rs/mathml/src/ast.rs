use derive_new::new;
use std::fmt;

pub mod operator;

use operator::Operator;

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct Mi(pub String);

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct Mrow(pub Vec<MathExpression>);

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
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

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct Ci {
    pub r#type: Option<Type>,
    pub content: Box<MathExpression>,
    pub func_of: Option<Vec<Ci>>,
}

/// The MathExpression enum represents the corresponding element type in MathML 3
/// (https://www.w3.org/TR/MathML3/appendixa.html#parsing_MathExpression)
#[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Clone, Hash, Default, new)]
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
            MathExpression::Mo(op) => {
                write!(f, "{}", op)
            }
            MathExpression::Mrow(Mrow(elements)) => {
                for e in elements {
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
