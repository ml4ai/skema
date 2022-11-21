use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mi, Mn, Mo, Mrow, Msub},
    Operator,
};
use crate::ast::MathExpression::{Mfrac, Msqrt};

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
                if pre.args.len() >= pre.op.len() {
                    pre.op.push(Operator::Multiply); // deal with the invisible multiply operator
                }
                pre.args.push(Expr::Atom(Atom::Identifier(x.to_string())));
            }
            Mn(x) => {
                if pre.args.len() >= pre.op.len() {
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
                pre_exp.op.push(Operator::Other("".to_string()));
                for mut x in xs {
                    &x.to_expr(&mut pre_exp);
                }
                pre.args.push(Expr::Expression { op: pre_exp.op, args: pre_exp.args }.clone());
            }
            Msqrt(xs) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                };
                pre_exp.op.push(Operator::Sqrt);
                &xs.to_expr(&mut pre_exp);
                pre.args.push(Expr::Expression { op: pre_exp.op, args: pre_exp.args }.clone());
            }
            Mfrac(xs1, xs2) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                };
                pre_exp.op.push(Operator::Other("".to_string()));
                &xs1.to_expr(&mut pre_exp);
                pre_exp.op.push(Operator::Divide);
                &xs2.to_expr(&mut pre_exp);
                pre.args.push(Expr::Expression { op: pre_exp.op, args: pre_exp.args }.clone());
            }
            _ => {
                panic!("Unhandled type!");
            }
        }
    }
}

impl Expr {
    fn group_expr(mut self) {
        match self {
            Expr::Atom(_) => {}
            Expr::Expression { op, args } => {
                let mut md_op_idx: Vec<usize> = Vec::new();
                if op.len() > 1 {
                    for o in 0..=self.op.len() {
                        if self.op[o] == Operator::Multiply || self.op[o] == Operator::Divide {
                            md_op_idx.push(o);
                        }
                    }
                    let mut op_copy = op.clone();
                    let mut args_copy = args.clone();
                    // combine consecutive * / terms as a new expression term
                    let mut pre_moi = -1;
                    let mut new_exp = Expr::Expression {
                                op: Vec::<Operator>::new(),
                                args: Vec::<Expr>::new(),
                            };
                    for moi in md_op_idx {
                        if moi == -1 || (moi != -1 && moi - pre_moi > 1) {
                            new_exp = Expr::Expression {
                                op: Vec::<Operator>::new(),
                                args: Vec::<Expr>::new(),
                            };
                        }
                        pre_moi = moi;
                        match &mut new_exp {
                            Expr::Atom(_) => {}
                            Expr::Expression { op, args } => {
                                op.push(op_copy[moi].clone());
                                args.push(args_copy[moi].clone());
                            }
                        }
                    }
                }
            }
        }
    }
}

impl PreExp {
    fn group_expr(mut self) {
        let mut md_op_idx: Vec<usize> = Vec::new();
        if self.op.len() > 1 {
            for o in 0..=self.op.len() {
                if self.op[o] == Operator::Multiply || self.op[o] == Operator::Divide {
                    md_op_idx.push(o);
                }
            }
            // combine consecutive * / terms as a new expression term
        }


        match &self.args[0] {
            Expr::Atom(_) => {}
            Expr::Expression { op, args } => {}
        };
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
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
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
    pre_exp.op.push(Operator::Other("root".to_string()));
    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(op[2], Operator::Subtract);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
            match &args[2] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(op[2], Operator::Multiply);
                    assert_eq!(args[0], Expr::Atom(Atom::Number("4".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("c".to_string())));
                    assert_eq!(args[2], Expr::Atom(Atom::Identifier("d".to_string())));
                }
            }
            println!("Success!");
        }
    }
}

#[test]
#[ignore]
fn test_to_expr3() {
    let math_expression =
        Msqrt(Box::from(Mrow(vec![Mi("a".to_string()),
                                  Mo(Operator::Add),
                                  Mi("b".to_string())])));
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Sqrt);
            match &args[0] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Add);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
                }
            }
            println!("Success!");
        }
    }
}

#[test]
#[ignore]
fn test_to_expr4() {
    let math_expression =
        Mfrac(Box::from(Mrow(vec![Mi("a".to_string()),
                                  Mo(Operator::Add),
                                  Mi("b".to_string())])), Box::from(Mi("c".to_string())));
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Divide);
            match &args[0] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Add);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
                }
            }
            match &args[1] {
                Expr::Atom(x) => {
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("c".to_string())));
                }
                Expr::Expression { op, args } => {}
            }
            println!("Success!");
        }
    }
}

#[test]
#[ignore]
fn test_to_expr5() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mi("c".to_string()),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(op[2], Operator::Multiply);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
            assert_eq!(args[2], Expr::Atom(Atom::Identifier("c".to_string())));
            println!("Success!");
        }
    }
}