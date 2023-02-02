use std::fmt;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Operator {
    Add,
    Multiply,
    Equals,
    Divide,
    Subtract,
    Sqrt,
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
            Operator::Other(op) => write!(f, "{op}"),
        }
    }
}

/// The MathExpression enum represents the corresponding element type in MathML 3
/// (https://www.w3.org/TR/MathML3/appendixa.html#parsing_MathExpression)
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum MathExpression {
    Mi(String),
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
}

impl fmt::Display for MathExpression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MathExpression::Mi(identifier) => write!(f, "{}", identifier.to_string()),
            MathExpression::Mn(number) => write!(f, "{}", number.to_string()),
            MathExpression::Msup(base, superscript) => {
                write!(f, "{}_{}", base.to_string(), superscript.to_string())
            }
            MathExpression::Msub(base, subscript) => {
                write!(f, "{}_{}", base.to_string(), subscript.to_string())
            }
            _ => panic!("Unhandled case!"),
        }
    }
}

/// The Math struct represents the corresponding element type in MathML 3
/// (https://www.w3.org/TR/MathML3/appendixa.html#parsing_math)
#[derive(Debug, PartialEq)]
pub struct Math {
    pub content: Vec<MathExpression>,
}
