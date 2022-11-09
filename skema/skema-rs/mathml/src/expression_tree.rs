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
    Expression { op: Operator, args: Vec<Expr> },
}

impl MathExpression {
    fn to_expr(&self) -> Expr {
        match self {
            Mi(x) => Expr::Atom(Atom::Identifier(x.clone())),
            Mo(x) => Expr::Atom(Atom::Operator(x.clone())),
            Mrow(xs) => Expr::Expression {
                op: Operator::Add,
                args: Vec::<Expr>::new(),
            },
            _ => {
                panic!("Unhandled type!");
            }
        }
    }
}

#[test]
fn test_to_expr() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
    ]);
    assert_eq!(
        math_expression.to_expr(),
        Expr::Expression {
            op: Operator::Add,
            args: vec![
                Expr::Atom(Atom::Identifier("a".to_string())),
                Expr::Atom(Atom::Identifier("b".to_string()))
            ]
        }
    );
}
