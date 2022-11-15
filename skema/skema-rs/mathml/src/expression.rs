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

#[derive(Debug, PartialEq, Clone)]
struct PreExp {
    op: Operator,
    args: Vec<Expr>,
}

impl MathExpression {
    fn to_expr(self, mut pre: &mut PreExp) {
        match self {
            Mi(x) => {
                pre.args.push(Expr::Atom(Atom::Identifier(x.to_string())));
            }
            Mo(x) => {
                match x {
                    // Operator::Other(_) => { pre.op = x.clone(); }
                    _ => { pre.op = x.clone(); }
                }
            }
            Mrow(xs) => {
                let mut pre_exp = PreExp {
                    op: Operator::Other("".to_string()),
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

// TODO: Fix the test below.
#[test]
#[ignore]
fn test_to_expr() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
    ]);
    let mut pre_exp = PreExp {
        op: Operator::Other("root".to_string()),
        args: Vec::<Expr>::new(),
    };

    math_expression.to_expr(&mut pre_exp);
    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op, &Operator::Add);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
            println!("Success!");
        }
    }

    // assert_eq!(
    //     pre_exp.args[1],
    //     Expr::Expression {
    //         op: Operator::Add,
    //         args: vec![
    //             Expr::Atom(Atom::Identifier("a".to_string())),
    //             Expr::Atom(Atom::Identifier("b".to_string())),
    //         ],
    //     }
    // );
}
