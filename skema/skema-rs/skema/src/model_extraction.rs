use crate::config::Config;
use mathml::ast::{operator::Operator, Math};
pub use mathml::mml2pn::{ACSet, Term};
use petgraph::prelude::*;

use std::string::ToString;

// new imports
use mathml::ast::Ci;
use mathml::ast::MathExpression::Mo;
use mathml::ast::Type::Function;
use mathml::ast::{MathExpression, Mi, Mrow};
use mathml::parsers::first_order_ode::{flatten_mults, FirstOrderODE};
use mathml::parsers::math_expression_tree::MathExpressionTree;
use mathml::petri_net::recognizers::is_add_or_subtract_operator;
use neo4rs;
use neo4rs::{query, Error};
use std::sync::Arc;

#[allow(non_snake_case)]
pub async fn module_id2mathml_MET_ast(module_id: i64, config: Config) -> Vec<FirstOrderODE> {
    let mut core_dynamics_ast = Vec::<FirstOrderODE>::new();

    let core_id = find_pn_dynamics(module_id, config.clone()).await; // gives back list of function nodes that might contain the dynamics

    if core_id.is_empty() {
        let deriv = Ci {
            r#type: Some(Function),
            content: Box::new(MathExpression::Mi(Mi("temp".to_string()))),
            func_of: None,
        };
        let operate = Operator::Subtract;
        let rhs_arg = MathExpressionTree::Atom(MathExpression::Mi(Mi("temp".to_string())));
        let rhs = MathExpressionTree::Cons(operate, [rhs_arg].to_vec());
        let fo_eq = FirstOrderODE {
            lhs_var: deriv.clone(),
            func_of: [deriv.clone()].to_vec(), // just place holders for construction
            with_respect_to: deriv.clone(),    // just place holders for construction
            rhs,
        };
        core_dynamics_ast.push(fo_eq);
    } else {
        core_dynamics_ast = subgrapg2_core_dyn_MET_ast(core_id[0], config.clone())
            .await
            .unwrap();
    }

    core_dynamics_ast
}

pub async fn module_id2mathml_ast(module_id: i64, config: Config) -> Vec<Math> {
    let core_id = find_pn_dynamics(module_id, config.clone()).await; // gives back list of function nodes that might contain the dynamics

    // 5. pass id to subgrapg2_core_dyn to get core dynamics
    let core_dynamics_ast = subgraph2_core_dyn_ast(core_id[0], config.clone())
        .await
        .unwrap();

    // need to convert core_synamics_ast to a Math object to then run the PN converter on it
    let mut math_content = Vec::<Math>::new();
    for eq in core_dynamics_ast.iter() {
        let math_ast = Math {
            content: vec![MathExpression::Mrow(Mrow(eq.clone()))],
        };
        math_content.push(math_ast.clone());
    }

    math_content
}

// this function finds the core dynamics and returns a vector of
// node id's that meet the criteria for identification
#[allow(clippy::if_same_then_else)]
pub async fn find_pn_dynamics(module_id: i64, config: Config) -> Vec<i64> {
    let graph = subgraph2petgraph(module_id, config.clone()).await;
    // 1. find each function node
    let mut function_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels()[0] == *"Function" {
            function_nodes.push(node);
        }
    }
    // 2. check and make sure only expressions in function
    // 3. check number of expressions and decide off that
    let mut functions = Vec::<petgraph::Graph<neo4rs::Node, neo4rs::Relation>>::new();
    for i in 0..function_nodes.len() {
        // grab the subgraph of the given expression
        functions.push(subgraph2petgraph(graph[function_nodes[i]].id(), config.clone()).await);
    }
    // get a sense of the number of expressions in each function
    let mut func_counter = 0;
    let mut core_func = Vec::<usize>::new();
    for func in &functions {
        let mut expression_counter = 0;
        let mut primitive_counter = 0;
        for node in func.node_indices() {
            if func[node].labels()[0] == *"Expression" {
                expression_counter += 1;
            }
            if func[node].labels()[0] == *"Primitive" {
                if func[node].get::<String>("name").unwrap() == *"ast.Mult" {
                    primitive_counter += 1;
                } else if func[node].get::<String>("name").unwrap() == *"ast.Add" {
                    primitive_counter += 1;
                } else if func[node].get::<String>("name").unwrap() == *"ast.Sub" {
                    primitive_counter += 1;
                } else if func[node].get::<String>("name").unwrap() == *"ast.USub" {
                    primitive_counter += 1;
                }
            }
        }
        if expression_counter >= 3 && primitive_counter >= 12 {
            core_func.push(func_counter);
        }
        func_counter += 1;
    }
    // 4. get the id of functions with enough expressions
    let mut core_id = Vec::<i64>::new();
    for c_func in core_func.iter() {
        for node in functions[*c_func].node_indices() {
            if functions[*c_func][node].labels()[0] == *"Function" {
                core_id.push(functions[*c_func][node].id());
            }
        }
    }

    core_id
}

#[allow(non_snake_case)]
pub async fn subgrapg2_core_dyn_MET_ast(
    root_node_id: i64,
    config: Config,
) -> Result<Vec<FirstOrderODE>, Error> {
    // get the petgraph of the subgraph
    let graph = subgraph2petgraph(root_node_id, config.clone()).await;

    // find all the expressions
    let mut expression_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels()[0] == *"Expression" {
            expression_nodes.push(node);
        }
    }

    let mut core_dynamics = Vec::<FirstOrderODE>::new();

    // initialize vector to collect all expression wiring graphs
    for i in 0..expression_nodes.len() {
        // grab the wiring subgraph of the given expression
        let mut sub_w = subgraph_wiring(graph[expression_nodes[i]].id(), config.clone())
            .await
            .unwrap();
        if sub_w.node_count() > 3 {
            let expr = trim_un_named(&mut sub_w, config.clone()).await;
            let mut root_node = Vec::<NodeIndex>::new();
            for node_index in expr.node_indices() {
                if expr[node_index].labels()[0].clone() == *"Opo" {
                    root_node.push(node_index);
                }
            }
            if root_node.len() >= 2 {
                // println!("More than one Opo! Skipping Expression!");
            } else {
                core_dynamics.push(tree_2_MET_ast(expr, root_node[0]).unwrap());
            }
        }
    }

    Ok(core_dynamics)
}

pub async fn subgraph2_core_dyn_ast(
    root_node_id: i64,
    config: Config,
) -> Result<Vec<Vec<MathExpression>>, Error> {
    // get the petgraph of the subgraph
    let graph = subgraph2petgraph(root_node_id, config.clone()).await;

    // find all the expressions
    let mut expression_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels()[0] == *"Expression" {
            expression_nodes.push(node);
            // println!("Expression Nodes: {:?}", graph[node].clone().id);
        }
    }

    let mut core_dynamics = Vec::<Vec<MathExpression>>::new();

    // initialize vector to collect all expression wiring graphs
    for i in 0..expression_nodes.len() {
        // grab the wiring subgraph of the given expression
        let mut sub_w = subgraph_wiring(graph[expression_nodes[i]].id(), config.clone())
            .await
            .unwrap();
        if sub_w.node_count() > 3 {
            let expr = trim_un_named(&mut sub_w, config.clone()).await;
            let mut root_node = Vec::<NodeIndex>::new();
            for node_index in expr.node_indices() {
                if expr[node_index].labels()[0].clone() == *"Opo" {
                    root_node.push(node_index);
                }
            }
            if root_node.len() >= 2 {
            } else {
                core_dynamics.push(tree_2_ast(expr, root_node[0]).unwrap());
            }
        }
    }

    Ok(core_dynamics)
}

#[allow(non_snake_case)]
fn tree_2_MET_ast(
    graph: &mut petgraph::Graph<neo4rs::Node, neo4rs::Relation>,
    root_node: NodeIndex,
) -> Result<FirstOrderODE, Error> {
    let mut fo_eq_vec = Vec::<FirstOrderODE>::new();
    let _math_vec = Vec::<MathExpressionTree>::new();
    let mut lhs = Vec::<Ci>::new();
    if graph[root_node].labels()[0] == *"Opo" {
        // we first construct the derivative of the first node
        let deriv_name: &str = &graph[root_node].get::<String>("name").unwrap();
        // this will let us know if additional trimming is needed to handle the code implementation of the equations
        // let mut step_impl = false; this will be used for step implementaion for later
        // This is very bespoke right now
        // this check is for if it's leibniz notation or not, will need to expand as more cases are creating,
        // currently we convert to leibniz form
        if deriv_name[0..1].to_string() == "d" {
            let deriv = Ci {
                r#type: Some(Function),
                content: Box::new(MathExpression::Mi(Mi(deriv_name[1..2].to_string()))),
                func_of: None,
            };
            lhs.push(deriv);
        } else {
            // step_impl = true; this will be used for step implementaion for later
            let deriv = Ci {
                r#type: Some(Function),
                content: Box::new(MathExpression::Mi(Mi(deriv_name[0..1].to_string()))),
                func_of: None,
            };
            lhs.push(deriv);
        }
        for node in graph.neighbors_directed(root_node, Outgoing) {
            if graph[node].labels()[0].clone() == *"Primitive" {
                let operate = get_operator_MET(graph, node); // output -> Operator
                let rhs_arg = get_args_MET(graph, node); // output -> Vec<MathExpressionTree>
                let rhs = MathExpressionTree::Cons(operate, rhs_arg); // MathExpressionTree
                let rhs_flat = flatten_mults(rhs.clone());
                let fo_eq = FirstOrderODE {
                    lhs_var: lhs[0].clone(),
                    func_of: [lhs[0].clone()].to_vec(), // just place holders for construction
                    with_respect_to: lhs[0].clone(),    // just place holders for construction
                    rhs: rhs_flat,
                };
                fo_eq_vec.push(fo_eq);
            } else {
                println!("Error, expect RHS to have at least 1 primitive");
            }
        }
    }
    println!("FirstOrderODE: {:?}", fo_eq_vec[0].rhs.clone().to_string());
    Ok(fo_eq_vec[0].clone())
}

#[allow(non_snake_case)]
pub fn get_args_MET(
    graph: &petgraph::Graph<neo4rs::Node, neo4rs::Relation>,
    root_node: NodeIndex,
) -> Vec<MathExpressionTree> {
    let mut args = Vec::<MathExpressionTree>::new();
    let mut arg_order = Vec::<i64>::new();

    // idea construct a vector of edge labels for the arguments
    // construct vector of math expressions
    // reorganize based on edge labels vector

    // construct vecs
    for node in graph.neighbors_directed(root_node, Outgoing) {
        // first need to check for operator
        if graph[node].labels()[0].clone() == *"Primitive" {
            let operate = get_operator_MET(graph, node); // output -> Operator
            let rhs_arg = get_args_MET(graph, node); // output -> Vec<MathExpressionTree>
            let rhs = MathExpressionTree::Cons(operate, rhs_arg); // MathExpressionTree
            args.push(rhs.clone());
        } else {
            // asummption it is atomic
            let temp_string = graph[node].get::<String>("name").unwrap().clone();
            let arg2 = MathExpressionTree::Atom(MathExpression::Mi(Mi(temp_string.clone())));
            args.push(arg2.clone());
        }

        // construct order of args
        let x = graph
            .edge_weight(graph.find_edge(root_node, node).unwrap())
            .unwrap()
            .get::<i64>("index")
            .unwrap();
        arg_order.push(x);
    }

    // fix order of args
    let mut ordered_args = args.clone();

    for (i, ind) in arg_order.iter().enumerate() {
        // the ind'th element of order_args is the ith element of the unordered args
        if ordered_args.len() > *ind as usize {
            let _temp = std::mem::replace(&mut ordered_args[*ind as usize], args[i].clone());
        }
    }

    ordered_args
}

// this gets the operator from the node name
#[allow(non_snake_case)]
#[allow(clippy::if_same_then_else)]
pub fn get_operator_MET(
    graph: &petgraph::Graph<neo4rs::Node, neo4rs::Relation>,
    root_node: NodeIndex,
) -> Operator {
    let mut op = Vec::<Operator>::new();
    if graph[root_node].get::<String>("name").unwrap() == *"ast.Mult" {
        op.push(Operator::Multiply);
    } else if graph[root_node].get::<String>("name").unwrap() == *"ast.Add" {
        op.push(Operator::Add);
    } else if graph[root_node].get::<String>("name").unwrap() == *"ast.Sub" {
        op.push(Operator::Subtract);
    } else if graph[root_node].get::<String>("name").unwrap() == *"ast.USub" {
        op.push(Operator::Subtract);
    } else if graph[root_node].get::<String>("name").unwrap() == *"ast.Div" {
        op.push(Operator::Divide);
    } else {
        op.push(Operator::Other(
            graph[root_node].get::<String>("name").unwrap(),
        ));
    }
    op[0].clone()
}

fn tree_2_ast(
    graph: &mut petgraph::Graph<neo4rs::Node, neo4rs::Relation>,
    root_node: NodeIndex,
) -> Result<Vec<MathExpression>, Error> {
    let mut math_vec = Vec::<MathExpression>::new();

    if graph[root_node].labels()[0] == *"Opo" {
        // we first construct the derivative of the first node
        let deriv_name: &str = &graph[root_node].get::<String>("name").unwrap();
        // this will let us know if additional trimming is needed to handle the code implementation of the equations
        let mut step_impl = false;
        // This is very bespoke right now
        // this check is for if it's leibniz notation or not, will need to expand as more cases are creating,
        // currently we convert to leibniz form
        if deriv_name[0..1].to_string() == "d" {
            let deriv = MathExpression::Mfrac(
                Box::new(MathExpression::Mrow(Mrow(vec![
                    MathExpression::Mi(Mi(deriv_name[0..1].to_string())),
                    MathExpression::Mi(Mi(deriv_name[1..2].to_string())),
                ]))),
                Box::new(MathExpression::Mrow(Mrow(vec![
                    MathExpression::Mi(Mi(deriv_name[2..3].to_string())),
                    MathExpression::Mi(Mi(deriv_name[3..4].to_string())),
                ]))),
            );
            math_vec.push(deriv);
        } else {
            step_impl = true;
            let deriv = MathExpression::Mfrac(
                Box::new(MathExpression::Mrow(Mrow(vec![
                    MathExpression::Mi(Mi("d".to_string())),
                    MathExpression::Mi(Mi(deriv_name[0..1].to_string())),
                ]))),
                Box::new(MathExpression::Mrow(Mrow(vec![
                    MathExpression::Mi(Mi("d".to_string())),
                    MathExpression::Mi(Mi("t".to_string())),
                ]))),
            );
            math_vec.push(deriv);
        }
        // we also push an Mo('=') here before traversing the tree to parse the rhs
        math_vec.push(MathExpression::Mo(Operator::Equals));
        // now we walk through the tree to parse the rest
        let _rhs_eq = Vec::<MathExpression>::new();
        let mut first_op = Vec::<MathExpression>::new();

        // this only distributing if the multiplication is the first operator
        for node in graph.neighbors_directed(root_node, Outgoing) {
            if graph[node].labels()[0].clone() == *"Primitive"
                && graph[node].get::<String>("name").unwrap() != *"ast.USub"
            {
                first_op.push(get_operator(graph, node));
                let mut arg1 = get_args(graph, node);
                if graph[node].get::<String>("name").unwrap() == *"ast.Mult" {
                    let arg1_mult = is_multiple_terms(arg1[0].clone());
                    let arg2_mult = is_multiple_terms(arg1[1].clone());
                    if arg1_mult {
                        let arg_dist = distribute_args(arg1[1].clone(), arg1[0].clone());
                        math_vec.extend_from_slice(&arg_dist.clone());
                    } else if arg2_mult {
                        let arg_dist = distribute_args(arg1[0].clone(), arg1[1].clone());
                        math_vec.extend_from_slice(&arg_dist.clone());
                    } else {
                        math_vec.extend_from_slice(&arg1[0].clone());
                        math_vec.push(first_op[0].clone());
                        math_vec.extend_from_slice(&arg1[1].clone());
                    }
                } else {
                    // need to test for when we have USub next to mults in a term that might be needed
                    // these args should all be multiplications of each other, aka an individual term
                    // there for we just check for a usub and if it exists, we remove it and swap the operator + -> - or - -> +
                    let _new_arg = Vec::<MathExpression>::new();
                    let mut usub_exist0 = false;
                    let mut usub_exist1 = false;
                    let mut usub_idx = Vec::<i32>::new();
                    for (i, ent) in arg1[0].clone().iter().enumerate() {
                        if *ent == MathExpression::Mo(Operator::Other("ast.USub".to_string())) {
                            usub_exist0 = true;
                            usub_idx.push(i.try_into().unwrap());
                        }
                    }
                    for (i, ent) in arg1[1].clone().iter().enumerate() {
                        if *ent == MathExpression::Mo(Operator::Other("ast.USub".to_string())) {
                            usub_exist1 = true;
                            usub_idx.push(i.try_into().unwrap());
                        }
                    }
                    if usub_exist0 {
                        for id in usub_idx.clone().iter().rev() {
                            arg1[0].remove(*id as usize);
                        }
                        if first_op[0] == MathExpression::Mo(Operator::Add) {
                            first_op[0] = MathExpression::Mo(Operator::Subtract);
                        } else {
                            first_op[0] = MathExpression::Mo(Operator::Add);
                        }
                        math_vec.extend_from_slice(&arg1[0].clone());
                        math_vec.push(first_op[0].clone());
                        math_vec.extend_from_slice(&arg1[1].clone());
                    } else if usub_exist1 {
                        for id in usub_idx.clone().iter().rev() {
                            arg1[1].remove(*id as usize);
                        }
                        if first_op[0] == MathExpression::Mo(Operator::Add) {
                            first_op[0] = MathExpression::Mo(Operator::Subtract);
                        } else {
                            first_op[0] = MathExpression::Mo(Operator::Add);
                        }
                        math_vec.extend_from_slice(&arg1[0].clone());
                        math_vec.push(first_op[0].clone());
                        math_vec.extend_from_slice(&arg1[1].clone());
                    } else {
                        math_vec.extend_from_slice(&arg1[0].clone());
                        math_vec.push(first_op[0].clone());
                        math_vec.extend_from_slice(&arg1[1].clone());
                    }
                }
            } else {
                println!("Not supported or Trivial case");
            }
        }

        // we now need to handle the case where it's step implementation
        // we find the Mi of the state variable that doesn't have a multiplication next to it
        // (including only one, if at the end of the vec)
        // we then remove it and the one of the addition operators next to it
        if step_impl {
            let ref_name = deriv_name[0..1].to_string();
            for (idx, obj) in math_vec.clone().iter().enumerate() {
                if *obj == MathExpression::Mi(Mi(ref_name.clone())) {
                    // find each index of the state variable on the rhs
                    // check if there is a multiplication to the right or left
                    // if no multiplication then delete entry and all neighboring addition operators
                    // this should complete the transformation to a leibniz diff eq from a euler method
                    if math_vec[idx - 1_usize].clone() == MathExpression::Mo(Operator::Add)
                        || math_vec[idx - 1_usize].clone() == MathExpression::Mo(Operator::Equals)
                    {
                        // check right side of operator, but need to be wary of vec end
                        let mut idx_last = false;
                        if idx == math_vec.clone().len() - 1 {
                            idx_last = true;
                        }
                        if !idx_last
                            && math_vec[idx + 1_usize].clone() == MathExpression::Mo(Operator::Add)
                        {
                            // delete idx and neighboring Add's (left side might be equals, which is kept)
                            math_vec.remove(idx + 1_usize);
                            math_vec.remove(idx);
                            if math_vec[idx - 1_usize].clone()
                                != MathExpression::Mo(Operator::Equals)
                            {
                                math_vec.remove(idx - 1_usize);
                            }
                        } else if !idx_last
                            && math_vec[idx + 1_usize].clone()
                                == MathExpression::Mo(Operator::Subtract)
                        {
                            // delete idx and neighboring Add's (left side might be equals, which is kept)
                            math_vec.remove(idx + 1_usize);
                            math_vec.remove(idx);
                            if math_vec[idx - 1_usize].clone()
                                != MathExpression::Mo(Operator::Equals)
                            {
                                math_vec.remove(idx - 1_usize);
                            } else {
                                // this puts the deleted subtract back but on the end, which will be correct once flipped
                                math_vec.push(MathExpression::Mo(Operator::Subtract));
                            }
                        } else if idx_last {
                            // delete idx and neighboring Add to the left
                            // need to delete from the largest index to the smallest as to not mess up the indexing
                            math_vec.remove(idx);
                            math_vec.remove(idx - 1_usize);
                        }
                    }
                }
            }
        }
    } else {
        println!("Error! Starting node is not Opo!");
    }

    // do to how the expression tree in generated we need to reverse the order
    let mut reversed_final_math = Vec::<MathExpression>::new();
    let vec_len_temp = math_vec.clone().len();

    reversed_final_math.extend_from_slice(&math_vec.clone()[0..2]);

    for (i, j) in math_vec.clone().iter().rev().enumerate() {
        if i != vec_len_temp && i != (vec_len_temp - 1) && i != (vec_len_temp - 2) {
            reversed_final_math.push(j.clone());
        }
    }

    // we now need to remove all the multiplications since the PN converter uses adjacency to
    // determine multiplications

    for (i, term) in reversed_final_math.clone().iter().enumerate().rev() {
        if *term == MathExpression::Mo(Operator::Multiply) {
            reversed_final_math.remove(i);
        }
    }

    Ok(reversed_final_math)
}

// this function returns a bool of if an expression has multiple terms (conjuction of +/-'s)
fn is_multiple_terms(arg: Vec<MathExpression>) -> bool {
    let mut add_sub_index = 0;

    for (i, expression) in arg.iter().enumerate() {
        if is_add_or_subtract_operator(expression) {
            add_sub_index = i;
        }
    }
    add_sub_index != 0
}

// This function returns a vector of indicies of where the operators that seperate terms in an expression are
fn terms_indicies(arg: Vec<MathExpression>) -> Vec<i32> {
    let mut add_sub_index_vec = Vec::<i32>::new();

    for (i, expression) in arg.iter().enumerate() {
        if is_add_or_subtract_operator(expression) {
            add_sub_index_vec.push(i.try_into().unwrap());
        }
    }
    add_sub_index_vec
}

// the function takes two arguments and distributes them, made so the first argument is distributed over the second
fn distribute_args(
    arg1: Vec<MathExpression>,
    mut arg2: Vec<MathExpression>,
) -> Vec<MathExpression> {
    let mut arg_dist = Vec::<MathExpression>::new();

    let arg2_term_ind = terms_indicies(arg2.clone());

    // check if need to swap operator signs
    /* Is this running properly, not removing Mo(Other("'USub'")) in I equation in SIR */
    if arg1[0] == Mo(Operator::Other("USub".to_string())) {
        println!("USub dist happens"); // This is never running
        println!("Entry 1"); // operator starts at begining of arg2
        if arg2_term_ind[0] == 0 {
            for (i, ind) in arg2_term_ind.iter().enumerate() {
                if arg2[*ind as usize] == Mo(Operator::Add) {
                    arg2[*ind as usize] = Mo(Operator::Subtract);
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    }
                } else {
                    arg2[*ind as usize] = Mo(Operator::Add);
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    }
                }
            }
        } else {
            println!("Entry 2");
            // operator doesn't start at beginning so have to add it manually
            arg_dist.push(Mo(Operator::Subtract));
            arg_dist.extend_from_slice(&arg2.clone()[0..(arg2_term_ind[0] - 1) as usize]);
            let vec_len1 = arg1.clone().len() - 1;
            arg_dist.extend_from_slice(&arg1[1..vec_len1]);
            for (i, ind) in arg2_term_ind.iter().enumerate() {
                if arg2[*ind as usize] == Mo(Operator::Add) {
                    arg2[*ind as usize] = Mo(Operator::Subtract);
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    }
                } else {
                    arg2[*ind as usize] = Mo(Operator::Add);
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    }
                }
            }
        }
    } else {
        // don't have to swap operators
        if arg2_term_ind[0] == 0 {
            println!("Entry 3");
            for (i, ind) in arg2_term_ind.iter().enumerate() {
                if arg2[*ind as usize] == Mo(Operator::Add) {
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    }
                } else if (i + 1) != arg2_term_ind.len() {
                    arg_dist.extend_from_slice(
                        &arg2.clone()[(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                    );
                    arg_dist.extend_from_slice(&arg1.clone()); // check
                } else {
                    // last of the expression case
                    let vec_len = arg2.clone().len() - 1;
                    arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                    arg_dist.extend_from_slice(&arg1.clone()); // check
                }
            }
        } else {
            println!("Entry 4");
            // don't swap operators manual beginning push
            arg_dist.extend_from_slice(&arg2.clone()[0..(arg2_term_ind[0] - 1) as usize]);
            let vec_len1 = arg1.clone().len();
            arg_dist.extend_from_slice(&arg1[1..vec_len1]);
            for (i, ind) in arg2_term_ind.iter().enumerate() {
                if arg2[*ind as usize] == Mo(Operator::Add) {
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    }
                } else if (i + 1) != arg2_term_ind.len() {
                    arg_dist.extend_from_slice(
                        &arg2.clone()[(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                    );
                    arg_dist.extend_from_slice(&arg1.clone()); // check
                } else {
                    // last of the expression case
                    let vec_len = arg2.clone().len() - 1;
                    arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                    arg_dist.extend_from_slice(&arg1.clone()); // check
                }
            }
        }
    }
    arg_dist
}

// this will get the arguments for an operator/primitive of the graph, assumes it's binary
fn get_args(
    graph: &petgraph::Graph<neo4rs::Node, neo4rs::Relation>,
    root_node: NodeIndex,
) -> Vec<Vec<MathExpression>> {
    let mut op = Vec::<MathExpression>::new();
    // need to construct the vector of length 2 with temporary filling since we might get the second
    // element first and tuples can't be dynamically indexed in rust
    let temp_op = Vec::<MathExpression>::new();
    let mut args = vec![temp_op; 2];

    for (i, node) in graph.neighbors_directed(root_node, Outgoing).enumerate() {
        if graph[node].labels()[0].clone() == *"Primitive"
            && graph[node].get::<String>("name").unwrap() == *"USub"
        {
            op.push(get_operator(graph, node));
            for node1 in graph.neighbors_directed(node, Outgoing) {
                let temp_mi = MathExpression::Mi(Mi(graph[node1].get::<String>("name").unwrap()
                    [0..(graph[node1].get::<String>("name").unwrap().len() - 1_usize)]
                    .to_string()));
                args[i].push(op[0].clone());
                args[i].push(temp_mi.clone());
            }
        } else if graph[node].labels()[0] == *"Opi" || graph[node].labels()[0] == *"Literal" {
            let temp_mi = MathExpression::Mi(Mi(graph[node].get::<String>("name").unwrap()
                [0..(graph[node].get::<String>("name").unwrap().len() - 1_usize)]
                .to_string()));
            args[i].push(temp_mi.clone());
        } else {
            let n_args = get_args(graph, node);
            let mut temp_vec = Vec::<MathExpression>::new();
            temp_vec.extend_from_slice(&n_args[0]);
            temp_vec.push(get_operator(graph, node));
            temp_vec.extend_from_slice(&n_args[1]);
            args[i].extend_from_slice(&temp_vec.clone());
        }
    }

    args
}

// this gets the operator from the node name
fn get_operator(
    graph: &petgraph::Graph<neo4rs::Node, neo4rs::Relation>,
    root_node: NodeIndex,
) -> MathExpression {
    let mut op = Vec::<MathExpression>::new();
    if graph[root_node].get::<String>("name").unwrap() == *"ast.Mult" {
        op.push(Mo(Operator::Multiply));
    } else if graph[root_node].get::<String>("name").unwrap() == *"ast.Add" {
        op.push(Mo(Operator::Add));
    } else if graph[root_node].get::<String>("name").unwrap() == *"ast.Sub" {
        op.push(Mo(Operator::Subtract));
    } else if graph[root_node].get::<String>("name").unwrap() == *"ast.Div" {
        op.push(Mo(Operator::Divide));
    } else {
        op.push(Mo(Operator::Other(
            graph[root_node].get::<String>("name").unwrap(),
        )));
    }
    op[0].clone()
}

// this currently only works for un-named nodes that are not chained or have multiple incoming/outgoing edges
async fn trim_un_named(
    graph: &mut petgraph::Graph<neo4rs::Node, neo4rs::Relation>,
    config: Config,
) -> &mut petgraph::Graph<neo4rs::Node, neo4rs::Relation> {
    // first create a cloned version of the graph we can modify while iterating over it.

    let graph_call = Arc::new(config.graphdb_connection().await);

    // iterate over the graph and add a new edge to bypass the un-named nodes
    for node_index in graph.node_indices() {
        if graph[node_index].get::<String>("name").unwrap().clone() == *"un-named" {
            let mut bypass = Vec::<NodeIndex>::new();
            for node1 in graph.neighbors_directed(node_index, Incoming) {
                bypass.push(node1);
            }
            for node2 in graph.neighbors_directed(node_index, Outgoing) {
                bypass.push(node2);
            }
            // one incoming one outgoing
            if bypass.len() == 2 {
                // annoyingly have to pull the edge/Relation to insert into graph
                let mut edge_list = Vec::<neo4rs::Relation>::new();
                let query_string = format!(
                    "MATCH (n)-[r:Wire]->(m) WHERE id(n) = {} AND id(m) = {} RETURN r",
                    graph[bypass[0]].id(),
                    graph[node_index].id()
                );
                let mut result = graph_call.execute(query(&query_string[..])).await.unwrap();
                while let Ok(Some(row)) = result.next().await {
                    let edge: neo4rs::Relation = row.get("r").unwrap();
                    edge_list.push(edge);
                }
                // add the bypass edge
                for edge in edge_list {
                    graph.add_edge(bypass[0], bypass[1], edge);
                }
            } else if bypass.len() > 2 {
                // this operates on the assumption that there maybe multiple references to the port
                // (incoming arrows) but only one outgoing arrow, this seems to be the case based on
                // data too.

                let end_node_idx = bypass.len() - 1;
                for (i, _ent) in bypass[0..end_node_idx].iter().enumerate() {
                    // this iterates over all but the last entry in the bypass vec
                    let mut edge_list = Vec::<neo4rs::Relation>::new();
                    let query_string = format!(
                        "MATCH (n)-[r:Wire]->(m) WHERE id(n) = {} AND id(m) = {} RETURN r",
                        graph[bypass[i]].id(),
                        graph[node_index].id()
                    );
                    let mut result = graph_call.execute(query(&query_string[..])).await.unwrap();
                    while let Ok(Some(row)) = result.next().await {
                        let edge: neo4rs::Relation = row.get("r").unwrap();
                        edge_list.push(edge);
                    }

                    for edge in edge_list {
                        graph.add_edge(bypass[i], bypass[end_node_idx], edge);
                    }
                }
            }
        }
    }

    // now we perform a filter_map to remove the un-named nodes and only the bypass edge will remain to connect the nodes
    // we also remove the unpack node if it is present here as well
    for node_index in graph.node_indices().rev() {
        if graph[node_index].get::<String>("name").unwrap().clone() == *"un-named"
            || graph[node_index].get::<String>("name").unwrap().clone() == *"unpack"
        {
            graph.remove_node(node_index);
        }
    }

    graph
}

async fn subgraph_wiring(
    module_id: i64,
    config: Config,
) -> Result<petgraph::Graph<neo4rs::Node, neo4rs::Relation>, Error> {
    let mut node_list = Vec::<neo4rs::Node>::new();
    let mut edge_list = Vec::<neo4rs::Relation>::new();

    // Connect to Memgraph.
    let graph = Arc::new(config.graphdb_connection().await);
    // node query
    let mut result1 = graph
        .execute(
            query(
                "MATCH (n)-[*]->(m) WHERE id(n) = $id
        MATCH q = (l)<-[r:Wire]-(m)
        WITH reduce(output = [], m IN nodes(q) | output + m ) AS nodes1
        UNWIND nodes1 AS nodes2
        WITH DISTINCT nodes2
        return nodes2",
            )
            .param("id", module_id),
        )
        .await?;
    while let Ok(Some(row)) = result1.next().await {
        let node: neo4rs::Node = row.get("nodes2").unwrap();
        node_list.push(node);
    }
    // edge query
    let mut result2 = graph
        .execute(
            query(
                "MATCH (n)-[*]->(m) WHERE id(n) = $id
        MATCH q = (l)<-[r:Wire]-(m)
        WITH reduce(output = [], m IN relationships(q) | output + m ) AS edges1
        UNWIND edges1 AS edges2
        WITH DISTINCT edges2
        return edges2",
            )
            .param("id", module_id),
        )
        .await?;
    while let Ok(Some(row)) = result2.next().await {
        let edge: neo4rs::Relation = row.get("edges2").unwrap();
        edge_list.push(edge);
    }

    let mut graph: petgraph::Graph<neo4rs::Node, neo4rs::Relation> = Graph::new();

    // Add nodes to the petgraph graph and collect their indexes
    let mut nodes = Vec::<NodeIndex>::new();
    for node in node_list {
        let n1 = graph.add_node(node);
        nodes.push(n1);
    }

    // add the edges to the petgraph
    for edge in edge_list {
        let mut src = Vec::<NodeIndex>::new();
        let mut tgt = Vec::<NodeIndex>::new();
        for node_idx in &nodes {
            if graph[*node_idx].id() == edge.start_node_id() {
                src.push(*node_idx);
            }
            if graph[*node_idx].id() == edge.end_node_id() {
                tgt.push(*node_idx);
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0], tgt[0], edge);
        }
    }
    Ok(graph)
}

async fn subgraph2petgraph(
    module_id: i64,
    config: Config,
) -> petgraph::Graph<neo4rs::Node, neo4rs::Relation> {
    let (x, y) = get_subgraph(module_id, config.clone()).await.unwrap();

    // Create a petgraph graph
    let mut graph: petgraph::Graph<neo4rs::Node, neo4rs::Relation> = Graph::new();

    // Add nodes to the petgraph graph and collect their indexes
    let mut nodes = Vec::<NodeIndex>::new();
    for node in x {
        let n1 = graph.add_node(node);
        nodes.push(n1);
    }

    // add the edges to the petgraph
    for edge in y {
        let mut src = Vec::<NodeIndex>::new();
        let mut tgt = Vec::<NodeIndex>::new();
        for node_idx in &nodes {
            if graph[*node_idx].id() == edge.start_node_id() {
                src.push(*node_idx);
            }
            if graph[*node_idx].id() == edge.end_node_id() {
                tgt.push(*node_idx);
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0], tgt[0], edge);
        }
    }
    graph
}

pub async fn get_subgraph(
    module_id: i64,
    config: Config,
) -> Result<(Vec<neo4rs::Node>, Vec<neo4rs::Relation>), Error> {
    // construct the query that will delete the module with a given unique identifier

    let mut node_list = Vec::<neo4rs::Node>::new();
    let mut edge_list = Vec::<neo4rs::Relation>::new();

    // Connect to Memgraph.
    let graph = Arc::new(config.graphdb_connection().await);
    // node query
    let mut result1 = graph
        .execute(
            query(
                "MATCH p = (n)-[r*]->(m) WHERE id(n) = $id
        WITH reduce(output = [], n IN nodes(p) | output + n ) AS nodes1
        UNWIND nodes1 AS nodes2
        WITH DISTINCT nodes2
        return nodes2",
            )
            .param("id", module_id),
        )
        .await?;
    while let Ok(Some(row)) = result1.next().await {
        let node: neo4rs::Node = row.get("nodes2").unwrap();
        node_list.push(node);
    }
    // edge query
    let mut result2 = graph
        .execute(
            query(
                "MATCH p = (n)-[r*]->(m) WHERE id(n) = $id
        WITH reduce(output = [], n IN relationships(p) | output + n ) AS edges1
        UNWIND edges1 AS edges2
        WITH DISTINCT edges2
        return edges2",
            )
            .param("id", module_id),
        )
        .await?;
    while let Ok(Some(row)) = result2.next().await {
        let edge: neo4rs::Relation = row.get("edges2").unwrap();
        edge_list.push(edge);
    }

    Ok((node_list, edge_list))
}
