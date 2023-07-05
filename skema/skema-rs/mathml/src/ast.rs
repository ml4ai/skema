use derive_new::new;
use std::fmt;

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
    /// Derivative operator, in line with Spivak notation: http://ceres-solver.org/spivak_notation.html
    Derivative {
        order: u8,
        var_index: u8,
    },
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
            Operator::Derivative { order, var_index } => write!(f, "D({order}, {var_index})"),
            Operator::Other(op) => write!(f, "{op}"),
        }
    }
}

#[derive(Debug, Ord, PartialOrd, PartialEq, Eq, Clone, Hash, new)]
pub struct Mi(pub String);

/// The MathExpression enum represents the corresponding element type in MathML 3
/// (https://www.w3.org/TR/MathML3/appendixa.html#parsing_MathExpression)
#[derive(Debug, PartialOrd, Ord, PartialEq, Eq, Clone, Hash, Default, new)]
pub enum MathExpression {
    Mi(Mi),
    Mo(Operator),
    Mn(String),
    Msqrt(Box<MathExpression>),
    Mrow(Vec<MathExpression>),
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
    #[default]
    None,
}

impl fmt::Display for MathExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathExpression::Mi(Mi(identifier)) => write!(f, "{}", identifier),
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
            expression => write!(f, "{expression:?}"),
        }
    }
}

/// The Math struct represents the corresponding element type in MathML 3
/// (https://www.w3.org/TR/MathML3/appendixa.html#parsing_math)
#[derive(Debug, PartialEq, Clone)]
pub struct Math {
    pub content: Vec<MathExpression>,
}
