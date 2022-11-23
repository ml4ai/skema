use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mi, Mn, Mo, Mrow, Msub},
    Operator,
};
use crate::ast::MathExpression::{Mfrac, Msqrt};
use std::cmp::Reverse;

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
    fn group_expr(&mut self) {
        match self {
            Expr::Atom(_) => {}
            Expr::Expression { op, args } => {
                let mut removed_idx = Vec::new();
                let mut op_copy = op.clone();
                let mut args_copy = args.clone();
                if op.len() > 2 {
                    let mut start_idx: i32 = -1;
                    let mut end_idx: i32 = -1;
                    let mut new_exp = Expr::Expression {
                        op: vec![Operator::Other("".to_string())],
                        args: Vec::<Expr>::new(),
                    };
                    for o in 0..=op.len() - 1 {
                        if op[o] == Operator::Multiply || op[o] == Operator::Divide {
                            removed_idx.push(o.clone());
                            if start_idx == -1 {
                                start_idx = o as i32;
                                end_idx = o as i32;
                                match &mut new_exp {
                                    Expr::Atom(_) => {}
                                    Expr::Expression { op, args } => {
                                        op.push(op_copy[o].clone());
                                        args.push(args_copy[o - 1].clone());
                                        args.push(args_copy[o].clone());
                                    }
                                }
                            } else if o as i32 - end_idx == 1 {
                                end_idx = o as i32;
                                match &mut new_exp {
                                    Expr::Atom(_) => {}
                                    Expr::Expression { op, args } => {
                                        op.push(op_copy[o].clone());
                                        args.push(args_copy[o].clone())
                                    }
                                }
                            } else {
                                args[start_idx as usize - 1] = new_exp.clone();
                                new_exp = Expr::Expression {
                                    op: vec![Operator::Other("".to_string())],
                                    args: Vec::<Expr>::new(),
                                };
                                match &mut new_exp {
                                    Expr::Atom(_) => {}
                                    Expr::Expression { op, args } => {
                                        op.push(op_copy[o].clone());
                                        args.push(args_copy[o - 1].clone());
                                        args.push(args_copy[o].clone());
                                    }
                                }
                                start_idx = o as i32;
                                end_idx = o as i32;
                            }
                        }
                    }

                    if removed_idx.len() == op.len() - 1 {
                        return;
                    }

                    match &mut new_exp {
                        Expr::Atom(_) => {}
                        Expr::Expression { op, .. } => {
                            if op.len() != 0 {
                                args[start_idx as usize - 1] = new_exp.clone();
                            }
                        }
                    }
                    for ri in removed_idx.iter().rev() {
                        op.remove(*ri);
                        args.remove(*ri);
                    }
                }

                for arg in args {
                    match arg {
                        Expr::Atom(_) => {}
                        Expr::Expression { .. } => { arg.group_expr(); }
                    }
                }
            }
        }
    }
}

impl PreExp {
    fn group_expr(&mut self) {
        for arg in &mut self.args {
            match arg {
                Expr::Atom(_) => {}
                Expr::Expression { .. } => { arg.group_expr(); }
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
    pre_exp.group_expr();
    println!();
    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            match &args[1] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("b".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("c".to_string())));
                }
            }
            println!("Success!");
        }
    }
}

#[test]
#[ignore]
fn test_to_expr6() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mi("c".to_string()),
        Mo(Operator::Multiply),
        Mi("d".to_string()),
        Mo(Operator::Divide),
        Mi("e".to_string()),
        Mo(Operator::Subtract),
        Mi("f".to_string()),
        Mo(Operator::Multiply),
        Mi("g".to_string()),
        Mo(Operator::Subtract),
        Mi("h".to_string()),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    println!();
    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(op[2], Operator::Subtract);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[3], Expr::Atom(Atom::Identifier("h".to_string())));
            match &args[1] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(op[2], Operator::Multiply);
                    assert_eq!(op[3], Operator::Divide);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("b".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("c".to_string())));
                    assert_eq!(args[2], Expr::Atom(Atom::Identifier("d".to_string())));
                    assert_eq!(args[3], Expr::Atom(Atom::Identifier("e".to_string())));
                }
            }
            match &args[2] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("f".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("g".to_string())));
                }
            }
            println!("Success!");
        }
    }
}