use crate::ast::{
    MathExpression,
    MathExpression::{Mfrac, Mi, Mn, Mo, Mrow, Msqrt},
    Operator,
};


use petgraph::visit::NodeRef;
use petgraph::{graph::NodeIndex, Graph};

use std::collections::VecDeque;




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
    pub fn to_expr(self, pre: &mut PreExp) {
        match self {
            Mi(x) => {
                if pre.args.len() >= pre.op.len() {
                    // deal with the invisible multiply operator
                    pre.op.push(Operator::Multiply);
                }
                pre.args.push(Expr::Atom(Atom::Identifier(x)));
            }
            Mn(x) => {
                if pre.args.len() >= pre.op.len() {
                    // deal with the invisible multiply operator
                    pre.op.push(Operator::Multiply);
                }
                pre.args.push(Expr::Atom(Atom::Number(x)));
            }
            Mo(x) => {
                {
                    pre.op.push(x);
                }
            }
            Mrow(xs) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                };
                pre_exp.op.push(Operator::Other("".to_string()));
                for x in xs {
                    x.to_expr(&mut pre_exp);
                }
                pre.args.push(
                    Expr::Expression {
                        op: pre_exp.op,
                        args: pre_exp.args,
                        name: "".to_string(),
                    }
                        ,
                );
            }
            Msqrt(xs) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                };
                pre_exp.op.push(Operator::Sqrt);
                xs.to_expr(&mut pre_exp);
                pre.args.push(
                    Expr::Expression {
                        op: pre_exp.op,
                        args: pre_exp.args,
                        name: "".to_string(),
                    }
                        ,
                );
            }
            Mfrac(xs1, xs2) => {
                let mut pre_exp = PreExp {
                    op: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                };
                pre_exp.op.push(Operator::Other("".to_string()));
                xs1.to_expr(&mut pre_exp);
                pre_exp.op.push(Operator::Divide);
                xs2.to_expr(&mut pre_exp);
                pre.args.push(
                    Expr::Expression {
                        op: pre_exp.op,
                        args: pre_exp.args,
                        name: "".to_string(),
                    }
                        ,
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
        
        pre_exp.to_graph()
    }
}

impl Expr {
    fn group_expr(&mut self) {
        match self {
            Expr::Atom(_) => {}
            Expr::Expression { op, args, .. } => {
                let mut removed_idx = Vec::new();
                let op_copy = op.clone();
                let args_copy = args.clone();
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
                            removed_idx.push(o);
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
                            if !op.is_empty() {
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
                "".to_string()
            }
            Expr::Expression { op, args, name } => {
                if op[0] == Operator::Other("".to_string()) && !all_multi_div(op) && !redundant_paren(name) {
                    name.push('(');
                    add_paren = true;
                }
                for i in 0..=op.len() - 1 {
                    if i > 0 {
                        if op[i] == Operator::Equals {
                            let mut remove_idx = Vec::new();
                            let mut x: i32 = (name.chars().count() - 1) as i32;
                            if x > 0 {
                                while x >= 0 {
                                    if name.chars().nth(x as usize) != Some('(') && name.chars().nth(x as usize) != Some(')') {
                                        remove_idx.push(x as usize);
                                    }
                                    x -= 1;
                                }
                            }
                            for i in remove_idx.iter() {
                                name.remove(*i);
                            }
                        } else {
                            name.push_str(&op[i].to_string().clone());
                        }
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
                            let mut string;
                            if op[0] != Operator::Other("".to_string()) {
                                string = op[0].to_string();
                                string.push_str(args[i].get_names().as_str().clone());
                            } else {
                                string = args[i].get_names().as_str().to_string().clone();
                            }
                            name.push_str(&string.clone());
                        }
                    }
                }
                if add_paren {
                    name.push(')');
                }
                add_paren = false;

                name.to_string()
            }
        }
    }

    fn to_graph(&mut self, graph: &mut MathExpressionGraph) {
        match self {
            Expr::Atom(_x) => {}
            Expr::Expression { op, args, name } => {
                let parent_node_index: NodeIndex = get_node_idx(graph, name);
                if op[0] != Operator::Other("".to_string()) {
                    let mut unitary_name = op[0].to_string();
                    let mut name_copy = name.to_string();
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
                                    if op_copy.len() > 1 {
                                        graph.add_edge(
                                            node_idx,
                                            parent_node_index,
                                            op_copy[i + 1].to_string(),
                                        );
                                    }
                                } else if op_copy[i] == Operator::Equals {
                                    if i <= op_copy.len() - 2 {
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
                            }
                            Atom::Identifier(x) => {
                                let node_idx = get_node_idx(graph, x);
                                if i == 0 {
                                    if op_copy.len() > 1 {
                                        graph.add_edge(
                                            node_idx,
                                            parent_node_index,
                                            op_copy[i + 1].to_string(),
                                        );
                                    }
                                } else if op_copy[i] == Operator::Equals {
                                    if i <= op_copy.len() - 2 {
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
                            }
                            Atom::Operator(_x) => {}
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
                                } else if op_copy[i] == Operator::Equals {
                                    if i <= op_copy.len() - 2 {
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
                                    }
                                } else if op_copy[i] == Operator::Equals {
                                    if i <= op_copy.len() - 2 {
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
                            }
                            args[i].to_graph(graph);
                        }
                    }
                }
            }
        }
    }
}

pub fn redundant_paren(string: &String) -> bool {
    let str_len = string.chars().count();
    if !string.starts_with('(') || string.chars().nth(str_len - 1) != Some(')') {
        return false;
    }
    let mut par_stack = VecDeque::new();
    par_stack.push_back("left_par");
    for i in 1..=str_len - 2 {
        if string.chars().nth(i) == Some('(') {
            par_stack.push_back("par");
        } else if string.chars().nth(i) == Some(')') {
            par_stack.pop_back();
        }
    }
    if !par_stack.is_empty() && par_stack[0] == "left_par" {
        return true;
    }
    false
}

pub fn all_multi_div(op: &mut Vec<Operator>) -> bool {
    for o in 1..=op.len() - 1 {
        if op[o] != Operator::Multiply && op[o] != Operator::Divide {
            return false;
        }
    }
    true
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
        g
    }
}

pub fn remove_paren(string: &mut String) -> &mut String {
    while redundant_paren(string) {
        string.remove(string.len() - 1);
        string.remove(0);
    }
    string
}

/// if exists, return the node index; if no, add a new node and return the node index
pub fn get_node_idx(graph: &mut MathExpressionGraph, name: &mut String) -> NodeIndex {
    remove_paren(name);
    if graph.node_count() > 0 {
        for n in 0..=graph.node_count() - 1 {
            match graph.raw_nodes().get(n) {
                None => {}
                Some(x) => {
                    if *name == x.weight {
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
    
    graph.add_node(name.to_string())
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
}

#[test]
fn test_to_expr17() {
    let math_expression = Mrow(vec![
        Mi("s".to_string()),
        Mo(Operator::Equals),
        Mi("b".to_string()),
        Mo(Operator::Multiply),
        Mrow(vec![
            Mi("a".to_string()),
            Mo(Operator::Subtract),
            Mi("b".to_string()),
        ]),
    ]);
    let g = math_expression.to_graph();
}

#[test]
fn test_to_expr18() {
    let math_expression = Mrow(vec![
        Mi("s".to_string()),
        Mo(Operator::Equals),
        Mi("a".to_string()),
        Mo(Operator::Multiply),
        Mi("b".to_string()),
        Mo(Operator::Subtract),
        Msqrt(Box::from(Mrow(vec![
            Mi("a".to_string()),
            Mo(Operator::Subtract),
            Mi("b".to_string()),
            Mo(Operator::Multiply),
            Mrow(vec![
                Mi("a".to_string()),
                Mo(Operator::Subtract),
                Mi("b".to_string()),
            ]),
        ]))),
    ]);
    let g = math_expression.to_graph();
}
