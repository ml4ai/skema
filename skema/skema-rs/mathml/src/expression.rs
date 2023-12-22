use crate::ast::{operator::Operator, MathExpression, Mi};
use crate::parsers::math_expression_tree::MathExpressionTree;
use petgraph::{graph::NodeIndex, Graph};
use std::{clone::Clone, collections::VecDeque};

/// Struct for representing mathematical expressions in order to align with source code.
pub type MathExpressionGraph<'a> = Graph<String, String>;

use petgraph::dot::Dot;
use std::string::ToString;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Atom {
    Number(String),
    Identifier(String),
    Operator(Operator),
}

/// Intermediate data structure to support the generation of graphs of mathematical expressions
#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Atom(Atom),
    Expression {
        ops: Vec<Operator>,
        args: Vec<Expr>,
        name: String,
    },
}

fn is_unary_operator(op: &Operator) -> bool {
    match op {
        Operator::Sqrt
        | Operator::Factorial
        | Operator::Exp
        | Operator::Grad
        | Operator::Div
        | Operator::Abs
        | Operator::Derivative(_)
        | Operator::Sin
        | Operator::Cos
        | Operator::Tan
        | Operator::Sec
        | Operator::Csc
        | Operator::Cot
        | Operator::Arcsin
        | Operator::Arccos
        | Operator::Arctan
        | Operator::Arcsec
        | Operator::Arccsc
        | Operator::Arccot
        | Operator::Mean => true,
        _ => false,
    }
}

/// Processes a MathExpression under the type of MathExpressionTree::Atom and appends
/// the corresponding LaTeX representation to the provided String.
fn process_atom_expression(expr: &MathExpression, expression: &mut Expr) {
    match expr {
        // If it's a Ci variant, recursively process its content
        MathExpression::Ci(x) => {
            process_atom_expression(&x.content, expression);
        }
        MathExpression::Mi(Mi(id)) => {
            if let Expr::Expression { ops, args, name } = expression {
                args.push(Expr::Atom(Atom::Identifier(id.replace(' ', ""))));
            }
        }
        MathExpression::Mn(number) => {
            if let Expr::Expression { ops, args, name } = expression {
                args.push(Expr::Atom(Atom::Number(number.replace(' ', ""))));
            }
        }
        MathExpression::Msqrt(x) => {
            let mut new_expr = Expr::Expression {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: String::new(),
            };
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Sqrt);
                process_atom_expression(x, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = expression {
                args.push(new_expr.clone());
            }
        }
        MathExpression::Mfrac(x1, x2) => {
            let mut new_expr = Expr::Expression {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: String::new(),
            };
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("".to_string()));
                process_atom_expression(x1, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Divide);
                process_atom_expression(x2, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = expression {
                args.push(new_expr.clone());
            }
        }
        MathExpression::Msup(x1, x2) => {
            let mut new_expr = Expr::Expression {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: String::new(),
            };
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("".to_string()));
                process_atom_expression(x1, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("^".to_string()));
                process_atom_expression(x2, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = expression {
                args.push(new_expr.clone());
            }
        }
        MathExpression::Msub(x1, x2) => {
            let mut new_expr = Expr::Expression {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: String::new(),
            };
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("".to_string()));
                process_atom_expression(x1, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("_".to_string()));
                process_atom_expression(x2, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = expression {
                args.push(new_expr.clone());
            }
        }
        MathExpression::Msubsup(x1, x2, x3) => {
            let mut new_expr = Expr::Expression {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: String::new(),
            };
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("".to_string()));
                process_atom_expression(x1, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("_".to_string()));
                process_atom_expression(x2, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("^".to_string()));
                process_atom_expression(x3, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = expression {
                args.push(new_expr.clone());
            }
        }
        MathExpression::Munder(x1, x2) => {
            let mut new_expr = Expr::Expression {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: String::new(),
            };
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("".to_string()));
                process_atom_expression(x1, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("under".to_string()));
                process_atom_expression(x2, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = expression {
                args.push(new_expr.clone());
            }
        }
        MathExpression::Mover(x1, x2) => {
            let mut new_expr = Expr::Expression {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: String::new(),
            };
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("".to_string()));
                process_atom_expression(x1, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("over".to_string()));
                process_atom_expression(x2, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = expression {
                args.push(new_expr.clone());
            }
        }
        MathExpression::Mtext(x) => {
            if let Expr::Expression { ops, args, name } = expression {
                args.push(Expr::Atom(Atom::Identifier(x.replace(' ', ""))));
            }
        }
        MathExpression::Mspace(x) => {
            if let Expr::Expression { ops, args, name } = expression {
                args.push(Expr::Atom(Atom::Identifier(x.to_string())));
            }
        }
        MathExpression::AbsoluteSup(x1, x2) => {
            let mut new_expr = Expr::Expression {
                ops: Vec::<Operator>::new(),
                args: Vec::<Expr>::new(),
                name: String::new(),
            };
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("|.|".to_string()));
                process_atom_expression(x1, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = &mut new_expr {
                ops.push(Operator::Other("_".to_string()));
                process_atom_expression(x2, &mut new_expr);
            }
            if let Expr::Expression { ops, args, name } = expression {
                args.push(new_expr.clone());
            }
        }
        MathExpression::Mrow(vec_me) => {
            for me in vec_me.0.iter() {
                let mut new_expr = Expr::Expression {
                    ops: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: String::new(),
                };
                if let Expr::Expression { ops, args, name } = &mut new_expr {
                    process_atom_expression(me, &mut new_expr);
                }
                if let Expr::Expression { ops, args, name } = expression {
                    args.push(new_expr.clone());
                }
            }
        }
        t => panic!("Unhandled MathExpression: {:?}", t),
    }
}

impl MathExpressionTree {
    /// Convert a MathExpressionTree struct to a Expression struct.
    pub fn to_expr(self, expr: &mut Expr) -> &mut Expr {
        match self {
            MathExpressionTree::Atom(a) => {
                process_atom_expression(&a, expr);
            }
            MathExpressionTree::Cons(head, rest) => {
                let mut new_expr = Expr::Expression {
                    ops: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: String::new(),
                };
                if is_unary_operator(&head) || (head == Operator::Subtract && rest.len() == 1) {
                    if let Expr::Expression { ops, args, name } = &mut new_expr {
                        ops.push(head);
                        rest[0].clone().to_expr(&mut new_expr);
                    }
                } else {
                    if let Expr::Expression { ops, args, name } = &mut new_expr {
                        ops.push(Operator::Other("".to_string()));
                        for (index, r) in rest.iter().enumerate() {
                            if index < rest.len() - 1 {
                                ops.push(head.clone());
                            }
                        }
                    }
                    if let Expr::Expression { ops, args, name } = &mut new_expr {
                        for r in &rest {
                            r.clone().to_expr(&mut new_expr);
                        }
                    }
                }
                if let Expr::Expression { ops, args, name } = expr {
                    args.push(new_expr.clone());
                }
            }
        }
        expr
    }
    pub fn to_graph(self) -> MathExpressionGraph<'static> {
        let mut expr = self.clone();
        let mut pre_exp = Expr::Expression {
            ops: vec![Operator::Other("root".to_string())],
            args: Vec::<Expr>::new(),
            name: "root".to_string(),
        };

        expr.to_expr(&mut pre_exp);

        if let Expr::Expression { ops, args, name } = &mut pre_exp {
            for mut arg in args {
                if let Expr::Expression { .. } = arg {
                    arg.group_expr();
                }
            }
        }
        if let Expr::Expression { ops, args, name } = &mut pre_exp {
            for mut arg in args {
                if let Expr::Expression { .. } = arg {
                    arg.collapse_expr();
                }
            }
        }
        /// if need to convert to canonical form, please uncomment the following
        // if let Expr::Expression {ops, args, name} = &mut pre_exp {
        //     for mut arg in args {
        //         if let Expr::Expression { .. } = arg {
        //             arg.distribute_expr();
        //         }
        //     }
        // }
        // if let Expr::Expression {ops, args, name} = &mut pre_exp {
        //     for mut arg in args {
        //         if let Expr::Expression { .. } = arg {
        //             arg.group_expr();
        //         }
        //     }
        // }
        // if let Expr::Expression {ops, args, name} = &mut pre_exp {
        //     for mut arg in args {
        //         if let Expr::Expression { .. } = arg {
        //             arg.collapse_expr();
        //         }
        //     }
        // }
        if let Expr::Expression { ops, args, name } = &mut pre_exp {
            for mut arg in args {
                if let Expr::Expression { .. } = arg {
                    arg.set_name();
                }
            }
        }
        let mut g = MathExpressionGraph::new();
        if let Expr::Expression { ops, args, name } = &mut pre_exp {
            for mut arg in args {
                if let Expr::Expression { .. } = arg {
                    arg.to_graph(&mut g);
                }
            }
        }
        g
    }
}

impl Expr {
    /// Group expression by multiplication and division operations.
    pub fn group_expr(&mut self) {
        if let Expr::Expression { ops, args, .. } = self {
            let mut indices_to_remove = Vec::new();
            let ops_copy = ops.clone();
            let args_copy = args.clone();
            if ops.len() > 2 {
                let mut start_index: i32 = -1;
                let mut end_index: i32 = -1;
                let mut new_exp = Expr::Expression {
                    ops: vec![Operator::Other("".to_string())],
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                };
                for (o, operator) in ops.iter().enumerate() {
                    if *operator == Operator::Multiply || *operator == Operator::Divide {
                        indices_to_remove.push(o);
                        if start_index == -1 {
                            start_index = o as i32;
                            end_index = o as i32;
                            if let Expr::Expression { ops, args, .. } = &mut new_exp {
                                ops.push(operator.clone());
                                args.push(args_copy[o - 1].clone());
                                args.push(args_copy[o].clone());
                            }
                        } else if o as i32 - end_index == 1 {
                            end_index = o as i32;
                            if let Expr::Expression { ops, args, .. } = &mut new_exp {
                                ops.push(ops_copy[o].clone());
                                args.push(args_copy[o].clone())
                            }
                        } else {
                            args[start_index as usize - 1] = new_exp.clone();
                            new_exp = Expr::Expression {
                                ops: vec![Operator::Other("".to_string())],
                                args: Vec::<Expr>::new(),
                                name: "".to_string(),
                            };
                            if let Expr::Expression { ops, args, .. } = &mut new_exp {
                                ops.push(ops_copy[o].clone());
                                args.push(args_copy[o - 1].clone());
                                args.push(args_copy[o].clone());
                            }
                            start_index = o as i32;
                            end_index = o as i32;
                        }
                    }
                }

                if indices_to_remove.len() == ops.len() - 1 {
                    return;
                }

                if let Expr::Expression { ops, .. } = &mut new_exp {
                    if !ops.is_empty() && start_index > 0 {
                        args[start_index as usize - 1] = new_exp.clone();
                    }
                }
                for ri in indices_to_remove.iter().rev() {
                    ops.remove(*ri);
                    args.remove(*ri);
                }
            }

            for arg in args {
                if let Expr::Expression { .. } = arg {
                    arg.group_expr();
                }
            }
        }
    }

    /// 1) If the current term's operators are all multiplication or division, check if it contains
    /// nested all multiplication or division terms inside. If so, collapse them into a single term.
    /// 2) If the current term's operators are all summation or subtraction, check if it contains
    /// nested all summation or subtraction terms inside. If so, collapse them into a single term.
    fn collapse_expr(&mut self) {
        if let Expr::Expression { ops, args, .. } = self {
            let mut ops_copy = ops.clone();
            let mut args_copy = args.clone();
            let mut shift;
            if all_ops_are_mult_or_div(ops.to_vec()) && ops.len() > 1 {
                let mut changed = true;
                while changed {
                    shift = 0;
                    for i in 0..args.len() {
                        let mut is_div: bool = false;
                        if ops[i] == Operator::Divide {
                            is_div = true;
                        };
                        if let Expr::Expression { ops, args, name: _ } = &mut args[i] {
                            if ops[0] == Operator::Other("".to_string())
                                && all_ops_are_mult_or_div(ops.to_vec())
                            {
                                args_copy[i + shift] = args[0].clone();
                                for j in 1..ops.len() {
                                    if is_div {
                                        ops_copy.insert(
                                            i + shift + j,
                                            switch_mul_div(ops[j].clone()).clone(),
                                        );
                                    } else {
                                        ops_copy.insert(i + shift + j, ops[j].clone());
                                    }
                                    args_copy.insert(i + shift + j, args[j].clone());
                                }
                                shift = shift + ops.len() - 1;
                            }
                        }
                    }
                    if ops.clone() == ops_copy.clone() {
                        changed = false;
                    }
                    *ops = ops_copy.clone();
                    *args = args_copy.clone();
                }
            }

            if all_ops_are_add_or_subt(ops.to_vec()) && ops.len() > 1 {
                let mut changed = true;
                while changed {
                    shift = 0;
                    for i in 0..args.len() {
                        let mut is_subt: bool = false;
                        if ops[i] == Operator::Subtract {
                            is_subt = true;
                        };
                        if let Expr::Expression { ops, args, name: _ } = &mut args[i] {
                            if ops[0] == Operator::Other("".to_string())
                                && all_ops_are_add_or_subt(ops.to_vec())
                            {
                                args_copy[i + shift] = args[0].clone();
                                for j in 1..ops.len() {
                                    if is_subt {
                                        ops_copy.insert(
                                            i + shift + j,
                                            switch_add_subt(ops[j].clone()).clone(),
                                        );
                                    } else {
                                        ops_copy.insert(i + shift + j, ops[j].clone());
                                    }
                                    args_copy.insert(i + shift + j, args[j].clone());
                                }
                                shift = shift + ops.len() - 1;
                            }
                        }
                    }

                    if ops.clone() == ops_copy.clone() {
                        changed = false;
                    }
                    *ops = ops_copy.clone();
                    *args = args_copy.clone();
                }
            }

            for arg in args {
                arg.collapse_expr();
            }
        }
    }

    /// 1) distribute variables and terms over multiplications, e.g., a*(b+c) => a*b+a*c
    /// 2) distribute variables and terms over divisions, e.g., a/(b+c)/(e+f) => a/(be+bf+ce+cf)
    #[allow(dead_code)] // used in tests I believe
    fn distribute_expr(&mut self) {
        if let Expr::Expression { ops, args, .. } = self {
            let mut ops_copy = ops.clone();
            let mut args_copy = args.clone();
            let mut distributed_ops: Vec<Operator> = Default::default();
            let mut distributed_terms: Vec<Expr> = Default::default();
            let mut is_distributed: bool = false;
            if ops_contain_mult(ops.to_vec()) && ops.len() > 1 {
                for i in 0..args.len() {
                    if i == 0 || ops[i] == Operator::Multiply {
                        if let Expr::Expression { ops, args, name: _ } = &mut args[i] {
                            if need_to_distribute(ops.to_vec()) {
                                is_distributed = true;
                                for j in 0..args.len() {
                                    let mut distributed_ops_unit = ops_copy.clone();
                                    let mut distributed_terms_unit = args_copy.clone();
                                    if j == 0 {
                                        distributed_ops.extend(distributed_ops_unit);
                                    } else {
                                        distributed_ops_unit[0] = ops[j].clone();
                                        distributed_ops.extend(distributed_ops_unit);
                                    }
                                    distributed_terms_unit[i] = args[j].clone();
                                    distributed_terms.extend(distributed_terms_unit);
                                }
                            }
                        }
                    }
                    if is_distributed {
                        let mut new_expr = Expr::Expression {
                            ops: distributed_ops.clone(),
                            args: distributed_terms.clone(),
                            name: "".to_string(),
                        };
                        new_expr.group_expr();
                        new_expr.collapse_expr();
                        new_expr.distribute_expr();
                        new_expr.group_expr();
                        new_expr.collapse_expr();
                        if let Expr::Expression { ops, args, name: _ } = &mut new_expr {
                            ops_copy = ops.clone();
                            args_copy = args.clone();
                        }
                        *ops = ops_copy;
                        *args = args_copy;
                        break;
                    }
                }
            }

            if need_to_distribute_divs(ops.to_vec(), args.to_vec()) && ops.len() > 1 {
                let mut new_expr = Expr::Expression {
                    ops: Vec::<Operator>::new(),
                    args: Vec::<Expr>::new(),
                    name: "".to_string(),
                };
                if let Expr::Expression {
                    ops,
                    args: _,
                    name: _,
                } = &mut new_expr
                {
                    ops.push(Operator::Other("".to_string()));
                }
                let mut removed_idx: Vec<usize> = Vec::new();
                for i in 0..args.len() {
                    if ops[i] == Operator::Divide {
                        removed_idx.push(i);
                        let tmp_arg = args[i].clone();
                        if let Expr::Expression { ops, args, name: _ } = &mut new_expr {
                            if ops.len() == args.len() {
                                ops.push(Operator::Multiply);
                            }
                            args.push(tmp_arg.clone());
                        }
                    }
                }
                new_expr.group_expr();
                new_expr.collapse_expr();
                new_expr.distribute_expr();
                new_expr.group_expr();
                new_expr.collapse_expr();

                for ri in removed_idx.iter().rev() {
                    ops.remove(*ri);
                    args.remove(*ri);
                }

                ops.push(Operator::Divide);
                args.push(new_expr);
            }

            for arg in args {
                arg.distribute_expr();
            }
        }
    }

    /// Construct a string representation of the Expression and store it under its 'name' property.
    pub fn set_name(&mut self) -> String {
        let mut add_paren = false;
        match self {
            Expr::Atom(_) => "".to_string(),
            Expr::Expression { ops, args, name } => {
                if ops[0] == Operator::Other("".to_string())
                    && !all_ops_are_mult_or_div(ops.to_vec())
                    && !contains_redundant_parens(name)
                {
                    name.push('(');
                    add_paren = true;
                }
                for i in 0..=ops.len() - 1 {
                    if i > 0 {
                        if ops[i] == Operator::Equals {
                            let mut new_name = "".to_string();
                            for n in name.as_bytes() {
                                if *n == 40_u8 {
                                    new_name.push('(');
                                }
                                if *n == 41_u8 {
                                    new_name.push(')');
                                }
                            }
                            *name = remove_redundant_parens(&mut new_name).clone();
                        } else {
                            name.push_str(&ops[i].to_string().clone());
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
                        Expr::Expression { ops, .. } => {
                            let mut string = "".to_string();
                            if ops[0] != Operator::Other("".to_string()) {
                                string = ops[0].to_string();
                                string.push('(');
                                string.push_str(args[i].set_name().as_str());
                                string.push(')');
                            } else {
                                string = args[i].set_name().as_str().to_string();
                            }
                            name.push_str(&string);
                        }
                    }
                }
                if add_paren {
                    name.push(')');
                }

                name.to_string()
            }
        }
    }

    pub fn to_graph(&mut self, graph: &mut MathExpressionGraph) {
        if let Expr::Expression { ops, args, name } = self {
            if name == "place_holder" {
                return;
            } else if name.contains("place_holder") {
                *name = name.replace("place_holder", "");
            }

            let mut parent_node_index: NodeIndex = Default::default();
            if ops[0].to_string() != "derivative" {
                parent_node_index = get_node_idx(graph, name)
            }
            let mut eq_loc = 0;
            if ops.contains(&Operator::Equals) {
                eq_loc = ops.iter().position(|r| r == &(Operator::Equals)).unwrap();
                let mut left_eq_name: String = "".to_string();
                for arg in args.iter_mut().take(eq_loc) {
                    match arg {
                        Expr::Atom(x) => match x {
                            Atom::Number(y) => {
                                left_eq_name.push_str(y);
                            }
                            Atom::Identifier(y) => {
                                left_eq_name.push_str(y);
                            }
                            Atom::Operator(_y) => {}
                        },
                        Expr::Expression { ops, args: _, name } => {
                            if ops[0] != Operator::Other("".to_string()) {
                                let mut unitary_name = ops[0].to_string();
                                let mut name_copy = name.to_string();
                                remove_redundant_parens(&mut name_copy);
                                unitary_name.push('(');
                                unitary_name.push_str(&name_copy);
                                unitary_name.push(')');
                                left_eq_name.push_str(unitary_name.as_str());
                            } else {
                                left_eq_name.push_str(name.as_str());
                            }
                        }
                    }
                }

                let node_idx = get_node_idx(graph, &mut left_eq_name);
                graph.update_edge(node_idx, parent_node_index, "=".to_string());
            }
            if ops[0] != Operator::Other("".to_string()) {
                let mut unitary_name = ops[0].to_string();
                let mut name_copy = name.to_string();
                remove_redundant_parens(&mut name_copy);
                unitary_name.push('(');
                unitary_name.push_str(&name_copy);
                unitary_name.push(')');
                let node_idx = get_node_idx(graph, &mut unitary_name);
                if ops[0].to_string() == "derivative" {
                    return;
                } else {
                    graph.update_edge(parent_node_index, node_idx, ops[0].to_string());
                }
            }
            let ops_copy = ops.clone();
            for i in eq_loc..=ops_copy.len() - 1 {
                match &mut args[i] {
                    Expr::Atom(x) => match x {
                        Atom::Number(x) => {
                            if x == "place_holder" {
                                continue;
                            } else if x.contains("place_holder") {
                                *x = x.replace("place_holder", "");
                            }
                            let node_idx = get_node_idx(graph, x);
                            if i == 0 {
                                if ops_copy.len() > 1 {
                                    if (ops_copy[i + 1].to_string() == "+"
                                        || ops_copy[i + 1].to_string() == "-")
                                        && !x.starts_with('-')
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "+".to_string(),
                                        );
                                    } else if ops_copy[i + 1].to_string() == "*"
                                        || ops_copy[i + 1].to_string() == "/"
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "*".to_string(),
                                        );
                                    } else {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            ops_copy[i + 1].to_string(),
                                        );
                                    }
                                }
                            } else if ops_copy[i] == Operator::Equals {
                                if i <= ops_copy.len() - 2 {
                                    if (ops_copy[i + 1].to_string() == "+"
                                        || ops_copy[i + 1].to_string() == "-")
                                        && !x.starts_with('-')
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "+".to_string(),
                                        );
                                    } else if ops_copy[i + 1].to_string() == "*"
                                        || ops_copy[i + 1].to_string() == "/"
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "*".to_string(),
                                        );
                                    } else {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            ops_copy[i + 1].to_string(),
                                        );
                                    }
                                }
                            } else {
                                graph.update_edge(
                                    node_idx,
                                    parent_node_index,
                                    ops_copy[i].to_string(),
                                );
                            }
                        }
                        Atom::Identifier(x) => {
                            if x == "place_holder" {
                                continue;
                            } else if x.contains("place_holder") {
                                *x = x.replace("place_holder", "");
                            }
                            let node_idx = get_node_idx(graph, x);
                            if i == 0 {
                                if ops_copy.len() > 1 {
                                    if (ops_copy[i + 1].to_string() == "+"
                                        || ops_copy[i + 1].to_string() == "-")
                                        && !x.starts_with('-')
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "+".to_string(),
                                        );
                                    } else if ops_copy[i + 1].to_string() == "*"
                                        || ops_copy[i + 1].to_string() == "/"
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "*".to_string(),
                                        );
                                    } else {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            ops_copy[i + 1].to_string(),
                                        );
                                    }
                                }
                            } else if ops_copy[i] == Operator::Equals {
                                if i <= ops_copy.len() - 2 {
                                    if (ops_copy[i + 1].to_string() == "+"
                                        || ops_copy[i + 1].to_string() == "-")
                                        && !x.starts_with('-')
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "+".to_string(),
                                        );
                                    } else if ops_copy[i + 1].to_string() == "*"
                                        || ops_copy[i + 1].to_string() == "/"
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "*".to_string(),
                                        );
                                    } else {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            ops_copy[i + 1].to_string(),
                                        );
                                    }
                                }
                            } else {
                                graph.update_edge(
                                    node_idx,
                                    parent_node_index,
                                    ops_copy[i].to_string(),
                                );
                            }
                        }
                        Atom::Operator(_x) => {}
                    },
                    Expr::Expression { ops, name, .. } => {
                        if name == "place_holder" {
                            continue;
                        } else if name.contains("place_holder") {
                            *name = name.replace("place_holder", "");
                        }
                        if ops[0] == Operator::Other("".to_string()) {
                            let node_idx = get_node_idx(graph, name);
                            if i == 0 {
                                if ops_copy.len() > 1 {
                                    if (ops_copy[i + 1].to_string() == "+"
                                        || ops_copy[i + 1].to_string() == "-")
                                        && !name.starts_with('-')
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "+".to_string(),
                                        );
                                    } else if ops_copy[i + 1].to_string() == "*"
                                        || ops_copy[i + 1].to_string() == "/"
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "*".to_string(),
                                        );
                                    } else {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            ops_copy[i + 1].to_string(),
                                        );
                                    }
                                }
                            } else if ops_copy[i] == Operator::Equals {
                                if i <= ops_copy.len() - 2 && ops_copy.len() > 1 {
                                    if (ops_copy[i + 1].to_string() == "+"
                                        || ops_copy[i + 1].to_string() == "-")
                                        && !name.starts_with('-')
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "+".to_string(),
                                        );
                                    } else if ops_copy[i + 1].to_string() == "*"
                                        || ops_copy[i + 1].to_string() == "/"
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "*".to_string(),
                                        );
                                    } else {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            ops_copy[i + 1].to_string(),
                                        );
                                    }
                                }
                            } else {
                                graph.update_edge(
                                    node_idx,
                                    parent_node_index,
                                    ops_copy[i].to_string(),
                                );
                            }
                        } else {
                            let mut unitary_name = ops[0].to_string();
                            let mut name_copy = name.to_string();
                            remove_redundant_parens(&mut name_copy);
                            unitary_name.push('(');
                            unitary_name.push_str(&name_copy);
                            unitary_name.push(')');
                            let node_idx = get_node_idx(graph, &mut unitary_name);
                            if i == 0 {
                                if ops_copy.len() > 1 {
                                    if (ops_copy[i + 1].to_string() == "+"
                                        || ops_copy[i + 1].to_string() == "-")
                                        && !unitary_name.starts_with('-')
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "+".to_string(),
                                        );
                                    } else if ops_copy[i + 1].to_string() == "*"
                                        || ops_copy[i + 1].to_string() == "/"
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "*".to_string(),
                                        );
                                    } else {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            ops_copy[i + 1].to_string(),
                                        );
                                    }
                                }
                            } else if ops_copy[i] == Operator::Equals {
                                if i <= ops_copy.len() - 2 {
                                    if (ops_copy[i + 1].to_string() == "+"
                                        || ops_copy[i + 1].to_string() == "-")
                                        && !unitary_name.starts_with('-')
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "+".to_string(),
                                        );
                                    } else if ops_copy[i + 1].to_string() == "*"
                                        || ops_copy[i + 1].to_string() == "/"
                                    {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            "*".to_string(),
                                        );
                                    } else {
                                        graph.update_edge(
                                            node_idx,
                                            parent_node_index,
                                            ops_copy[i + 1].to_string(),
                                        );
                                    }
                                }
                            } else {
                                graph.update_edge(
                                    node_idx,
                                    parent_node_index,
                                    ops_copy[i].to_string(),
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

/// The graph generation process adds additional parentheses to preserve operation order. This
/// function checks for the existence of those redundant parentheses.
pub fn contains_redundant_parens(string: &str) -> bool {
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

/// Check if the current term's operators are all multiply or divide.
pub fn all_ops_are_mult_or_div(ops: Vec<Operator>) -> bool {
    for op in ops.iter().take((ops.len() - 1) + 1).skip(1) {
        if op != &Operator::Multiply && op != &Operator::Divide {
            return false;
        }
    }
    true
}

///Switch the multiplication operator and the division operator
pub fn switch_mul_div(op: Operator) -> Operator {
    let switched_op: Operator = Operator::Other("".to_string());
    if op == Operator::Multiply {
        return Operator::Divide;
    }
    if op == Operator::Divide {
        return Operator::Multiply;
    }
    switched_op
}

///Switch the summation operator and the subtraction operator
pub fn switch_add_subt(op: Operator) -> Operator {
    let switched_op: Operator = Operator::Other("".to_string());
    if op == Operator::Add {
        return Operator::Subtract;
    }
    if op == Operator::Subtract {
        return Operator::Add;
    }
    switched_op
}

/// Check if the current term's operators are all add or subtract.
pub fn all_ops_are_add_or_subt(ops: Vec<Operator>) -> bool {
    for op in ops.iter().take((ops.len() - 1) + 1).skip(1) {
        if op != &Operator::Add && op != &Operator::Subtract {
            return false;
        }
    }
    true
}

/// Check if the current term's operators contain multiply.
pub fn ops_contain_mult(ops: Vec<Operator>) -> bool {
    for op in ops.iter().take((ops.len() - 1) + 1).skip(1) {
        if op == &Operator::Multiply {
            return true;
        }
    }
    false
}

/// Check if the current term's operators contain multiple divisions, and the denominators contain
/// add and subtract and without unary operators
pub fn need_to_distribute_divs(ops: Vec<Operator>, args: Vec<Expr>) -> bool {
    let mut num_div: i32 = 0;
    let mut contain_add_subt_without_uop: bool = false;
    for o in 1..=ops.len() - 1 {
        if ops[o] == Operator::Divide {
            num_div += 1;
            if let Expr::Expression {
                ops,
                args: _,
                name: _,
            } = &args[o]
            {
                if ops[0] == Operator::Other("".to_string()) && all_ops_are_add_or_subt(ops.clone())
                {
                    contain_add_subt_without_uop = true;
                }
            }
            if num_div > 1 && contain_add_subt_without_uop {
                return true;
            }
        }
    }
    false
}

/// Check if the current term's operators contain add or minus and without the unary operator.
pub fn need_to_distribute(ops: Vec<Operator>) -> bool {
    if ops[0] != Operator::Other("".to_string()) {
        return false;
    }
    for op in ops.iter().take((ops.len() - 1) + 1).skip(1) {
        if op == &Operator::Add || op == &Operator::Subtract {
            return true;
        }
    }
    false
}

/// Remove redundant parentheses.
pub fn remove_redundant_parens(string: &mut String) -> &mut String {
    while contains_redundant_parens(string) {
        string.remove(string.len() - 1);
        string.remove(0);
    }
    *string = string.replace("()", "");
    string
}

/// Return the node index if it exists; if not, add a new node and return the node index.
pub fn get_node_idx(graph: &mut MathExpressionGraph, name: &mut String) -> NodeIndex {
    remove_redundant_parens(name);
    if name.contains("derivative") {
        *name = name.replace('/', ", ");
    }
    if graph.node_count() > 0 {
        for n in 0..=graph.node_count() - 1 {
            if let Some(x) = graph.raw_nodes().get(n) {
                if *name == x.weight {
                    if let Some(x) = graph.node_indices().nth(n) {
                        return x;
                    }
                }
            }
        }
    }

    graph.add_node(name.to_string())
}

#[test]
fn test_plus_to_graph() {
    let input = "
    <math>
        <mrow>
            <mi>a</mi>
            <mo>+</mo>
            <mi>b</mi>
        </mrow>
    </math>
    ";
    let exp = input.parse::<MathExpressionTree>().unwrap();
    let g = exp.to_graph();
    let dot_representation = Dot::new(&g);
    assert_eq!(
        dot_representation
            .to_string()
            .replace("\n", "")
            .replace(" ", ""),
        "digraph{0[label=\"a+b\"]1[label=\"a\"]2[label=\"b\"]1->0[label=\"+\"]2->0[label=\"+\"]}"
    )
}

#[test]
fn test_equation_halfar_dome_8_1_to_graph() {
    let input = "
    <math>
      <mfrac>
        <mrow>
          <mi>&#x2202;</mi>
          <mi>H</mi>
        </mrow>
        <mrow>
          <mi>&#x2202;</mi>
          <mi>t</mi>
        </mrow>
      </mfrac>
      <mo>=</mo>
      <mi>&#x2207;</mi>
      <mo>&#x22C5;</mo>
      <mo>(</mo>
      <mi>&#x0393;</mi>
      <msup>
        <mi>H</mi>
        <mrow>
          <mi>n</mi>
          <mo>+</mo>
          <mn>2</mn>
        </mrow>
      </msup>
      <mo>|</mo>
      <mi>&#x2207;</mi>
      <mi>H</mi>
      <msup>
        <mo>|</mo>
        <mrow>
          <mi>n</mi>
          <mo>&#x2212;</mo>
          <mn>1</mn>
        </mrow>
      </msup>
      <mi>&#x2207;</mi>
      <mi>H</mi>
      <mo>)</mo>
    </math>
    ";

    let exp = input.parse::<MathExpressionTree>().unwrap();
    let g = exp.to_graph();
    let dot_representation = Dot::new(&g);
    assert_eq!(dot_representation.to_string()
                   .replace("\n", "")
                   .replace(" ", ""),
               "digraph{0[label=\"Div(Γ*(H^(n+2))*(Abs(Grad(H))^(n-1))*Grad(H))\"]1[label=\"D(1,t)(H)\"]2[label=\"Γ*(H^(n+2))*(Abs(Grad(H))^(n-1))*Grad(H)\"]3[label=\"Γ\"]4[label=\"H^(n+2)\"]5[label=\"H\"]6[label=\"n+2\"]7[label=\"n\"]8[label=\"2\"]9[label=\"Abs(Grad(H))^(n-1)\"]10[label=\"Abs(Grad(H))\"]11[label=\"Grad(H)\"]12[label=\"n-1\"]13[label=\"1\"]1->0[label=\"=\"]2->0[label=\"Div\"]3->2[label=\"*\"]4->2[label=\"*\"]5->4[label=\"^\"]6->4[label=\"^\"]7->6[label=\"+\"]8->6[label=\"+\"]9->2[label=\"*\"]10->9[label=\"^\"]11->10[label=\"Abs\"]5->11[label=\"Grad\"]12->9[label=\"^\"]7->12[label=\"+\"]13->12[label=\"-\"]11->2[label=\"*\"]}");
}

#[test]
fn test_equation_sidarthe_1_to_graph() {
    let input = "
    <math>
      <mrow>
        <mover>
          <mi>S</mi>
          <mo>&#x02D9;</mo>
        </mover>
      </mrow>
      <mo>(</mo>
      <mi>t</mi>
      <mo>)</mo>
      <mo>=</mo>
      <mo>&#x2212;</mo>
      <mi>S</mi>
      <mo>(</mo>
      <mi>t</mi>
      <mo>)</mo>
      <mo>(</mo>
      <mi>&#x03B1;</mi>
      <mi>I</mi>
      <mo>(</mo>
      <mi>t</mi>
      <mo>)</mo>
      <mo>+</mo>
      <mi>&#x03B2;</mi>
      <mi>D</mi>
      <mo>(</mo>
      <mi>t</mi>
      <mo>)</mo>
      <mo>+</mo>
      <mi>&#x03B3;</mi>
      <mi>A</mi>
      <mo>(</mo>
      <mi>t</mi>
      <mo>)</mo>
      <mo>+</mo>
      <mi>&#x03B4;</mi>
      <mi>R</mi>
      <mo>(</mo>
      <mi>t</mi>
      <mo>)</mo>
      <mo>)</mo>
    </math>
    ";

    let exp = input.parse::<MathExpressionTree>().unwrap();
    let g = exp.to_graph();
    let dot_representation = Dot::new(&g);
    assert_eq!(dot_representation.to_string()
                   .replace("\n", "")
                   .replace(" ", ""),
               "digraph{0[label=\"-(S)*(α*I+β*D+γ*A+δ*R)\"]1[label=\"D(1,t)(S)\"]2[label=\"-(S)\"]3[label=\"S\"]4[label=\"α*I+β*D+γ*A+δ*R\"]5[label=\"α*I\"]6[label=\"α\"]7[label=\"I\"]8[label=\"β*D\"]9[label=\"β\"]10[label=\"D\"]11[label=\"γ*A\"]12[label=\"γ\"]13[label=\"A\"]14[label=\"δ*R\"]15[label=\"δ\"]16[label=\"R\"]1->0[label=\"=\"]2->0[label=\"*\"]3->2[label=\"-\"]4->0[label=\"*\"]5->4[label=\"+\"]6->5[label=\"*\"]7->5[label=\"*\"]8->4[label=\"+\"]9->8[label=\"*\"]10->8[label=\"*\"]11->4[label=\"+\"]12->11[label=\"*\"]13->11[label=\"*\"]14->4[label=\"+\"]15->14[label=\"*\"]16->14[label=\"*\"]}");
}
