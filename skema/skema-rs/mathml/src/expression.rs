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
    Expression { op: Vec<Operator>, args: Vec<Expr> },
}

#[derive(Debug, PartialEq, Clone)]
struct PreExp {
    op: Vec<Operator>,
    args: Vec<Expr>,
}

impl MathExpression {
    fn to_expr(self, mut pre: &mut PreExp) {
        match self {
            Mi(x) => {
                if pre.args.len() > pre.op.len() {
                    pre.op.push(Operator::Multiply); // deal with the invisible multiply operator
                }
                pre.args.push(Expr::Atom(Atom::Identifier(x.to_string())));
            }
            Mn(x) => {
                if pre.args.len() > pre.op.len() {
                    pre.op.push(Operator::Multiply); // deal with the invisible multiply operator
                }
                pre.args.push(Expr::Atom(Atom::Number(x.to_string())));
            }
            Mo(x) => {
                match x {
                    // Operator::Other(_) => { pre.op = x.clone(); }
                    _ => { pre.op.push(x.clone()); }
                }
            }
            Mrow(xs) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                };
                for mut x in xs {
                    &x.to_expr(&mut pre_exp);
                }
                pre.args.push(Expr::Expression { op: pre_exp.op, args: pre_exp.args }.clone());
            }
            _ => {
                panic!("Unhandled type!");
            }
        }
    }
}

#[test]
#[ignore]
fn test_to_expr() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
    };

    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Add);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
            println!("Success!");
        }
    }
}

#[test]
#[ignore]
fn test_to_expr2() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Subtract),
        Mrow(vec![Mn("4".to_string()), Mi("c".to_string()), Mi("d".to_string())]),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
    };

    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {},
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Add);
            assert_eq!(op[1], Operator::Subtract);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
            match &args[2]{
                Expr::Atom(_) => {},
                Expr::Expression {op, args} => {
                    assert_eq!(op[0], Operator::Multiply);
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(args[0], Expr::Atom(Atom::Number("4".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("c".to_string())));
                    assert_eq!(args[2], Expr::Atom(Atom::Identifier("d".to_string())));
                }
            }
            println!("Success!");
        }
    }
}