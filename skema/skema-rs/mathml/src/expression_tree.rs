use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mi, Mn, Mo, Mrow, Msub},
    Operator,
};

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

impl MathExpression {
    fn to_expr(&self) -> Expr {
        match self {
            Mi(x) => Expr::Atom(Atom::Identifier(x.clone())),
            Mo(x) => Expr::Atom(Atom::Operator(x.clone())),
            Mrow(xs) => {
                Expr::Expression(Operator::Add)

            }
        }
    }
}

#[test]
fn test_to_expr() {}
