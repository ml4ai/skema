use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mi, Mn, Mo, Mrow, Msub},
};

#[derive(Debug, PartialEq, Clone)]
enum Operator {
    Add,
    Subtract,
    Equals,
    Other(String),
}

#[derive(Debug, PartialEq, Clone)]
enum Atom {
    Number(String),
    Identifier(String),
    Operator(Operator),
}

#[derive(Debug, PartialEq, Clone)]
enum Expr {
    Atom(Atom),
    Expression(Operator, Vec<Expr>),
}

//impl MathExpression {
//fn to_operator(&self) -> Option<Operator> {
//match self {
//Mo(x) => match x {
//"=".to_string() => Some(Operator::Equals),
//"+".to_string() => Some(Operator::Add),
//"-".to_string() => Some(Operator::Subtract),
//other => Some(Operator::Other(other)),
//},
//_ => None,
//}
//}
//}

#[test]
fn test_to_expr() {}
