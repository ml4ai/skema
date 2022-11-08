#[derive(Debug, PartialEq, Clone)]
pub enum MathExpression {
    Mi(String),
    Mo(String),
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
