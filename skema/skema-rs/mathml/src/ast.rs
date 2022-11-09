use std::fmt;

#[derive(Debug, PartialEq, Clone)]
pub enum Operator {
    Add,
    Subtract,
    Equals,
    // Catchall for operators we haven't explicitly defined as enum variants yet.
    Other(String),
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Add => write!(f, "{}", "+"),
            Operator::Subtract => write!(f, "{}", "-"),
            Operator::Equals => write!(f, "{}", "-"),
            Operator::Other(op) => write!(f, "{}", op),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum MathExpression {
    Mi(String),
    Mo(Operator),
    Mn(String),
    Msqrt(Box<MathExpression>),
    Mrow(Vec<MathExpression>),
    Mfrac(Box<MathExpression>, Box<MathExpression>),
    Msup(Box<MathExpression>, Box<MathExpression>),
    Msub(Box<MathExpression>, Box<MathExpression>),
    Munder(Vec<MathExpression>),
    Mover(Vec<MathExpression>),
    Msubsup(Vec<MathExpression>),
    Mtext(String),
    Mstyle(Vec<MathExpression>),
    Mspace(String),
    MoLine(String),
}

#[derive(Debug, PartialEq)]
pub struct Math {
    pub content: Vec<MathExpression>,
}
