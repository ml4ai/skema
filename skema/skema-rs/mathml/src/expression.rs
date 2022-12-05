use crate::ast::{
    Math, MathExpression,
    MathExpression::{Mfrac, Mi, Mn, Mo, Mrow, Msqrt, Msub},
    Operator,
};
use petgraph::dot::{Config, Dot};
use petgraph::graph::Node;
use petgraph::visit::NodeRef;
use petgraph::{graph::NodeIndex, Graph};
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::ptr::null_mut;

pub type MathExpressionGraph<'a> = Graph<String, String>;

#[derive(Debug, PartialEq, Clone)]
enum Atom {
    Number(String),
    Identifier(String),
    Operator(Operator),
}

#[derive(Debug, PartialEq, Clone)]
enum Expr {
    Atom(Atom),
    Expression {
        op: Vec<Operator>,
        args: Vec<Expr>,
        name: String,
    },
}

/// Intermediate data structure to support the generation of graphs of mathematical expressions
#[derive(Debug, PartialEq, Clone)]
pub struct PreExp {
    op: Vec<Operator>,
    args: Vec<Expr>,
    name: String,
}

impl MathExpression {
    pub fn to_expr(self, mut pre: &mut PreExp) {
        match self {
            Mi(x) => {
                if pre.args.len() >= pre.op.len() {
                    /// deal with the invisible multiply operator
                    pre.op.push(Operator::Multiply);
                }
                pre.args.push(Expr::Atom(Atom::Identifier(x.to_string())));
            }
            Mn(x) => {
                if pre.args.len() >= pre.op.len() {
                    /// deal with the invisible multiply operator
                    pre.op.push(Operator::Multiply);
                }
                pre.args.push(Expr::Atom(Atom::Number(x.to_string())));
            }
            Mo(x) => {
                match x {
                    _ => {
                        pre.op.push(x.clone());
                    }
                }
            }
            Mrow(xs) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                };
                pre_exp.op.push(Operator::Other("".to_string()));
                for mut x in xs {
                    &x.to_expr(&mut pre_exp);
                }
                pre.args.push(
                    Expr::Expression {
                        op: pre_exp.op,
                        args: pre_exp.args,
                        name: "".to_string(),
                    }
                    .clone(),
                );
            }
            Msqrt(xs) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                };
                pre_exp.op.push(Operator::Sqrt);
                &xs.to_expr(&mut pre_exp);
                pre.args.push(
                    Expr::Expression {
                        op: pre_exp.op,
                        args: pre_exp.args,
                        name: "".to_string(),
                    }
                    .clone(),
                );
            }
            Mfrac(xs1, xs2) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                };
                pre_exp.op.push(Operator::Other("".to_string()));
                &xs1.to_expr(&mut pre_exp);
                pre_exp.op.push(Operator::Divide);
                &xs2.to_expr(&mut pre_exp);
                pre.args.push(
                    Expr::Expression {
                        op: pre_exp.op,
                        args: pre_exp.args,
                        name: "".to_string(),
                    }
                    .clone(),
                );
            }
            _ => {
                panic!("Unhandled type!");
            }
        }
    }

    pub fn to_graph(self) -> MathExpressionGraph<'static> {
        let mut pre_exp = PreExp {
            op: Vec::<Operator>::new(),
            args: Vec::<Expr>::new(),
            name: "root".to_string(),
        };
        pre_exp.op.push(Operator::Other("root".to_string()));
        self.to_expr(&mut pre_exp);
        pre_exp.group_expr();
        pre_exp.get_names();
        let g = pre_exp.to_graph();
        return g;
    }
}

impl Expr {
    fn group_expr(&mut self) {
        match self {
            Expr::Atom(_) => {}
            Expr::Expression { op, args, .. } => {
                let mut removed_idx = Vec::new();
                let mut op_copy = op.clone();
                let mut args_copy = args.clone();
                if op.len() > 2 {
                    let mut start_idx: i32 = -1;
                    let mut end_idx: i32 = -1;
                    let mut new_exp = Expr::Expression {
                        op: vec![Operator::Other("".to_string())],
                        args: Vec::<Expr>::new(),
                        name: "".to_string(),
                    };
                    for o in 0..=op.len() - 1 {
                        if op[o] == Operator::Multiply || op[o] == Operator::Divide {
                            removed_idx.push(o.clone());
                            if start_idx == -1 {
                                start_idx = o as i32;
                                end_idx = o as i32;
                                match &mut new_exp {
                                    Expr::Atom(_) => {}
                                    Expr::Expression { op, args, .. } => {
                                        op.push(op_copy[o].clone());
                                        args.push(args_copy[o - 1].clone());
                                        args.push(args_copy[o].clone());
                                    }
                                }
                            } else if o as i32 - end_idx == 1 {
                                end_idx = o as i32;
                                match &mut new_exp {
                                    Expr::Atom(_) => {}
                                    Expr::Expression { op, args, .. } => {
                                        op.push(op_copy[o].clone());
                                        args.push(args_copy[o].clone())
                                    }
                                }
                            } else {
                                args[start_idx as usize - 1] = new_exp.clone();
                                new_exp = Expr::Expression {
                                    op: vec![Operator::Other("".to_string())],
                                    args: Vec::<Expr>::new(),
                                    name: "".to_string(),
                                };
                                match &mut new_exp {
                                    Expr::Atom(_) => {}
                                    Expr::Expression { op, args, .. } => {
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
                        Expr::Expression { .. } => {
                            arg.group_expr();
                        }
                    }
                }
            }
        }
    }

    fn get_names(&mut self) -> String {
        let mut add_paren = false;
        match self {
            Expr::Atom(_) => {
                return "".to_string();
            }
            Expr::Expression { op, args, name } => {
                if op[0] == Operator::Other("".to_string()) {
                    if !all_multi_div(op) {
                        if !redundant_paren(name) {
                            name.push_str("(");
                            add_paren = true;
                        }
                    }
                }
                for i in 0..=op.len() - 1 {
                    if i > 0 {
                        name.push_str(&op[i].to_string().clone());
                    }
                    match &mut args[i] {
                        Expr::Atom(x) => match x {
                            Atom::Number(x) => {
                                name.push_str(&x.to_string().clone());
                            }
                            Atom::Identifier(x) => {
                                name.push_str(&x.to_string().clone());
                            }
                            Atom::Operator(_) => {}
                        },
                        Expr::Expression { op, .. } => {
                            let mut str;
                            if op[0] != Operator::Other("".to_string()) {
                                str = op[0].to_string();
                                str.push_str(args[i].get_names().as_str().clone());
                            } else {
                                str = args[i].get_names().as_str().to_string().clone();
                            }
                            name.push_str(&str.clone());
                        }
                    }
                }
                if add_paren {
                    name.push_str(")");
                }
                add_paren = false;

                return name.to_string();
            }
        }
    }

    fn to_graph(&mut self, graph: &mut MathExpressionGraph) {
        match self {
            Expr::Atom(x) => {}
            Expr::Expression { op, args, name } => {
                let parent_node_index: NodeIndex = get_node_idx(graph, name);
                if op[0] != Operator::Other("".to_string()) {
                    let mut unitary_name = op[0].to_string();
                    let mut name_copy = name.to_string().clone();
                    remove_paren(&mut name_copy);
                    unitary_name.push_str("(".clone());
                    unitary_name.push_str(&name_copy.clone());
                    unitary_name.push_str(")".clone());
                    let node_idx = get_node_idx(graph, &mut unitary_name);
                    graph.add_edge(parent_node_index, node_idx, op[0].to_string());
                }
                let op_copy = op.clone();
                for i in 0..=op_copy.len() - 1 {
                    match &mut args[i] {
                        Expr::Atom(x) => match x {
                            Atom::Number(x) => {
                                let node_idx = get_node_idx(graph, x);
                                if i == 0 {
                                    graph.add_edge(
                                        node_idx,
                                        parent_node_index,
                                        op[i + 1].to_string(),
                                    );
                                } else {
                                    graph.add_edge(node_idx, parent_node_index, op[i].to_string());
                                }
                            }
                            Atom::Identifier(x) => {
                                let node_idx = get_node_idx(graph, x);
                                if i == 0 {
                                    graph.add_edge(
                                        node_idx,
                                        parent_node_index,
                                        op_copy[i + 1].to_string(),
                                    );
                                } else {
                                    graph.add_edge(
                                        node_idx,
                                        parent_node_index,
                                        op_copy[i].to_string(),
                                    );
                                }
                            }
                            Atom::Operator(x) => {}
                        },
                        Expr::Expression { op, name, .. } => {
                            if op[0] == Operator::Other("".to_string()) {
                                let node_idx = get_node_idx(graph, name);
                                if i == 0 {
                                    if op_copy.len() > 1 {
                                        graph.add_edge(
                                            node_idx,
                                            parent_node_index,
                                            op_copy[i + 1].to_string(),
                                        );
                                    }
                                } else {
                                    graph.add_edge(
                                        node_idx,
                                        parent_node_index,
                                        op_copy[i].to_string(),
                                    );
                                }
                            } else {
                                let mut unitary_name = op[0].to_string();
                                let mut name_copy = name.to_string().clone();
                                remove_paren(&mut name_copy);
                                unitary_name.push_str("(".clone());
                                unitary_name.push_str(&name_copy.clone());
                                unitary_name.push_str(")".clone());
                                let node_idx = get_node_idx(graph, &mut unitary_name);
                                if i == 0 {
                                    if op_copy.len() > 1 {
                                        graph.add_edge(
                                            node_idx,
                                            parent_node_index,
                                            op_copy[i + 1].to_string(),
                                        );
                                    } else {
                                        graph.add_edge(
                                            node_idx,
                                            parent_node_index,
                                            op[0].to_string(),
                                        );
                                    }
                                } else {
                                    graph.add_edge(
                                        node_idx,
                                        parent_node_index,
                                        op_copy[i].to_string(),
                                    );
                                }
                            }
                            args[i].to_graph(graph);
                        }
                    }
                }
            }
        }
    }
}

pub fn redundant_paren(str: &String) -> bool {
    let str_len = get_str_len(str);
    if str.chars().nth(0) != Some('(') || str.chars().nth(str_len - 1) != Some(')') {
        return false;
    }
    let mut par_stack = VecDeque::new();
    par_stack.push_back("left_par");
    for i in 1..=str_len - 2 {
        if str.chars().nth(i) == Some('(') {
            par_stack.push_back("par");
        } else if str.chars().nth(i) == Some(')') {
            par_stack.pop_back();
        }
    }
    if par_stack.len() > 0 {
        if par_stack[0] == "left_par" {
            return true;
        }
    }
    return false;
}

pub fn all_multi_div(op: &mut Vec<Operator>) -> bool {
    for o in 1..=op.len() - 1 {
        if op[o] != Operator::Multiply && op[o] != Operator::Divide {
            return false;
        }
    }
    return true;
}

impl PreExp {
    fn group_expr(&mut self) {
        for arg in &mut self.args {
            match arg {
                Expr::Atom(_) => {}
                Expr::Expression { .. } => {
                    arg.group_expr();
                }
            }
        }
    }

    fn get_names(&mut self) {
        for mut arg in &mut self.args {
            match &mut arg {
                Expr::Atom(_) => {}
                Expr::Expression { .. } => {
                    arg.get_names();
                }
            }
        }
    }

    fn to_graph(&mut self) -> MathExpressionGraph {
        let mut g = MathExpressionGraph::new();
        for mut arg in &mut self.args {
            match &mut arg {
                Expr::Atom(_) => {}
                Expr::Expression { .. } => {
                    arg.to_graph(&mut g);
                }
            }
        }
        return g;
    }
}

pub fn get_str_len(str: &str) -> usize {
    let mut count = 0;
    if str.len() > 0 {
        for i in 0..=str.len() - 1 {
            match str.chars().nth(i) {
                None => {}
                Some(_) => {
                    count = count + 1;
                }
            }
        }
    }
    return count;
}

pub fn remove_paren(str: &mut String) -> &mut String {
    while redundant_paren(str) {
        str.remove(str.len() - 1);
        str.remove(0);
    }
    return str;
}

/// if exists, return the node index; if no, add a new node and return the node index
pub fn get_node_idx(graph: &mut MathExpressionGraph, name: &mut String) -> NodeIndex {
    remove_paren(name);
    if graph.node_count() > 0 {
        for n in 0..=graph.node_count() - 1 {
            match graph.raw_nodes().get(n) {
                None => {}
                Some(x) => {
                    if name.to_string() == x.weight.to_string() {
                        match graph.node_indices().nth(n) {
                            None => {}
                            Some(x) => {
                                return x;
                            }
                        }
                    }
                }
            }
        }
    }
    let node_idx = graph.add_node(name.to_string());
    return node_idx;
}

#[test]
fn test_to_expr() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, .. } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
            println!("Success!");
        }
    }
}

#[test]
fn test_to_expr2() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Subtract),
        Mrow(vec![
            Mn("4".to_string()),
            Mi("c".to_string()),
            Mi("d".to_string()),
        ]),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "".to_string(),
    };

    math_expression.to_expr(&mut pre_exp);
    pre_exp.op.push(Operator::Other("root".to_string()));
    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, .. } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(op[2], Operator::Subtract);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[1], Expr::Atom(Atom::Identifier("b".to_string())));
            match &args[2] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, .. } => {
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
fn test_to_expr3() {
    let math_expression = Msqrt(Box::from(Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
    ])));
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, .. } => {
            assert_eq!(op[0], Operator::Sqrt);
            match &args[0] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, .. } => {
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
fn test_to_expr4() {
    let math_expression = Mfrac(
        Box::from(Mrow(vec![
            Mi("a".to_string()),
            Mo(Operator::Add),
            Mi("b".to_string()),
        ])),
        Box::from(Mi("c".to_string())),
    );
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, .. } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Divide);
            match &args[0] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, .. } => {
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
                Expr::Expression { op, args, .. } => {}
            }
            println!("Success!");
        }
    }
}

#[test]
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
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, .. } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            match &args[1] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, .. } => {
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
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, .. } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(op[2], Operator::Subtract);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[3], Expr::Atom(Atom::Identifier("h".to_string())));
            match &args[1] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, .. } => {
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
                Expr::Expression { op, args, .. } => {
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

#[test]
fn test_to_expr7() {
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
        name: "root".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, name } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(name, "(a+b*c)");
            match &args[1] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, name } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("b".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("c".to_string())));
                    assert_eq!(name, "b*c");
                }
            }
            println!("Success!");
        }
    }
}

#[test]
fn test_to_expr8() {
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
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, name } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(op[2], Operator::Subtract);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(args[3], Expr::Atom(Atom::Identifier("h".to_string())));
            assert_eq!(name, "(a+b*c*d/e-f*g-h)");
            match &args[1] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, name } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(op[2], Operator::Multiply);
                    assert_eq!(op[3], Operator::Divide);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("b".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("c".to_string())));
                    assert_eq!(args[2], Expr::Atom(Atom::Identifier("d".to_string())));
                    assert_eq!(args[3], Expr::Atom(Atom::Identifier("e".to_string())));
                    assert_eq!(name, "b*c*d/e");
                }
            }
            match &args[2] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, name } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("f".to_string())));
                    assert_eq!(args[1], Expr::Atom(Atom::Identifier("g".to_string())));
                    assert_eq!(name, "f*g");
                }
            }
            println!("Success!");
        }
    }
}

#[test]
fn test_to_expr9() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mrow(vec![
            Mi("c".to_string()),
            Mo(Operator::Subtract),
            Mi("d".to_string()),
        ]),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "root".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();

    match &pre_exp.args[0] {
        Expr::Atom(_) => {}
        Expr::Expression { op, args, name } => {
            assert_eq!(op[0], Operator::Other("".to_string()));
            assert_eq!(op[1], Operator::Add);
            assert_eq!(args[0], Expr::Atom(Atom::Identifier("a".to_string())));
            assert_eq!(name, "(a+b*(c-d))");
            match &args[1] {
                Expr::Atom(_) => {}
                Expr::Expression { op, args, name } => {
                    assert_eq!(op[0], Operator::Other("".to_string()));
                    assert_eq!(op[1], Operator::Multiply);
                    assert_eq!(args[0], Expr::Atom(Atom::Identifier("b".to_string())));
                    assert_eq!(name, "b*(c-d)");
                    match &args[1] {
                        Expr::Atom(_) => {}
                        Expr::Expression { op, args, name } => {
                            assert_eq!(op[0], Operator::Other("".to_string()));
                            assert_eq!(op[1], Operator::Subtract);
                            assert_eq!(args[0], Expr::Atom(Atom::Identifier("c".to_string())));
                            assert_eq!(args[1], Expr::Atom(Atom::Identifier("d".to_string())));
                            assert_eq!(name, "(c-d)");
                        }
                    }
                }
            }
            println!("Success!");
        }
    }
}

#[test]
fn test_to_expr10() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mrow(vec![
            Mi("c".to_string()),
            Mo(Operator::Subtract),
            Mi("a".to_string()),
        ]),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "root".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();
    let g = pre_exp.to_graph();
    println!("{}", Dot::new(&g));
}

#[test]
fn test_to_expr11() {
    let math_expression = Msqrt(Box::from(Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Subtract),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mrow(vec![
            Mi("a".to_string()),
            Mo(Operator::Subtract),
            Mi("b".to_string()),
        ]),
    ])));

    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "root".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();
    let g = pre_exp.to_graph();
    println!("{}", Dot::new(&g));
}

#[test]
fn test_to_expr12() {
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
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();
    let g = pre_exp.to_graph();
    println!("{}", Dot::new(&g));
}

#[test]
fn test_to_expr13() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mi("c".to_string()),
        Mo(Operator::Multiply),
        Mi("a".to_string()),
        Mo(Operator::Divide),
        Mi("d".to_string()),
        Mo(Operator::Subtract),
        Mi("c".to_string()),
        Mo(Operator::Multiply),
        Mi("a".to_string()),
        Mo(Operator::Subtract),
        Mi("b".to_string()),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();
    let g = pre_exp.to_graph();
    println!("{}", Dot::new(&g));
}

#[test]
fn test_to_expr14() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mi("c".to_string()),
        Mo(Operator::Multiply),
        Mrow(vec![
            Mi("a".to_string()),
            Mo(Operator::Subtract),
            Mi("b".to_string()),
        ]),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();
    let g = pre_exp.to_graph();
    println!("{}", Dot::new(&g));
}

#[test]
fn test_to_expr15() {
    let math_expression = Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Add),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mi("c".to_string()),
        Mo(Operator::Subtract),
        Msqrt(Box::from(Mrow(vec![
            Mi("a".to_string()),
            Mo(Operator::Add),
            Mi("b".to_string()),
        ]))),
    ]);
    let mut pre_exp = PreExp {
        op: Vec::<Operator>::new(),
        args: Vec::<Expr>::new(),
        name: "".to_string(),
    };
    pre_exp.op.push(Operator::Other("root".to_string()));
    math_expression.to_expr(&mut pre_exp);
    pre_exp.group_expr();
    pre_exp.get_names();
    let g = pre_exp.to_graph();
    println!("{}", Dot::new(&g));
}

#[test]
fn test_to_expr16() {
    let math_expression = Msqrt(Box::from(Mrow(vec![
        Mi("a".to_string()),
        Mo(Operator::Subtract),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mrow(vec![
            Mi("a".to_string()),
            Mo(Operator::Subtract),
            Mi("b".to_string()),
        ]),
    ])));
    let g = math_expression.to_graph();
    println!("{}", Dot::new(&g));
}
