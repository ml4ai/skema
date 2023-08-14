use mathml::ast::{operator::Operator, Math};

pub use mathml::mml2pn::{ACSet, Term};
use petgraph::prelude::*;
use rsmgclient::{ConnectParams, Connection, MgError, Node, Relationship, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::string::ToString;

// new imports
use mathml::ast::Ci;
use mathml::ast::MathExpression::Mo;
use mathml::ast::Type::Function;
use mathml::ast::{MathExpression, Mi, Mrow};
use mathml::parsers::first_order_ode::{flatten_mults, FirstOrderODE};
use mathml::parsers::math_expression_tree::MathExpressionTree;
use mathml::petri_net::recognizers::is_add_or_subtract_operator;

// struct for returning line spans
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize)]
pub struct LineSpan {
    line_begin: i64,
    line_end: i64,
}

// this function returns the line numbers of the function node id provided
pub fn get_line_span(
    node_id: i64,
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
) -> LineSpan {
    // extract line numbers of function which contains the dynamics
    let mut line_nums = Vec::<i64>::new();
    for node in graph.node_indices() {
        if graph[node].id == node_id {
            for n_node in graph.neighbors_directed(node, Outgoing) {
                if graph[n_node].labels == ["Metadata"] {
                    match &graph[n_node].properties["line_begin"] {
                        Value::List(x) => match x[0] {
                            Value::Int(y) => {
                                //println!("line_begin: {:?}", y);
                                line_nums.push(y);
                            }
                            _ => println!("error metadata type"),
                        },
                        _ => println!("error metadata type"),
                    }
                    match &graph[n_node].clone().properties["line_end"] {
                        Value::List(x) => match x[0] {
                            Value::Int(y) => {
                                //println!("line_end: {:?}", y);
                                line_nums.push(y);
                            }
                            _ => println!("error metadata type"),
                        },
                        _ => println!("error metadata type"),
                    }
                }
            }
        }
    }

    LineSpan {
        line_begin: line_nums[0],
        line_end: line_nums[1],
    }
}

pub fn module_id2mathml_MET_ast(module_id: i64, host: &str) -> Vec<FirstOrderODE> {
    let graph = subgraph2petgraph(module_id, host); // makes petgraph of graph

    let core_id = find_pn_dynamics(module_id, host); // gives back list of function nodes that might contain the dynamics

    let _line_span = get_line_span(core_id[0], graph); // get's the line span of function id

    //println!("\n{:?}", line_span);

    //println!("function_core_id: {:?}", core_id[0].clone());
    //println!("module_id: {:?}\n", module_id.clone());
    // 4.5 now to check if of those expressions, if they are arithmetric in nature

    // 5. pass id to subgrapg2_core_dyn to get core dynamics
    let (core_dynamics_ast, _metadata_map_ast) =
        subgrapg2_core_dyn_MET_ast(core_id[0], host).unwrap();

    core_dynamics_ast
}

pub fn module_id2mathml_ast(module_id: i64, host: &str) -> Vec<Math> {
    let graph = subgraph2petgraph(module_id, host); // makes petgraph of graph

    let core_id = find_pn_dynamics(module_id, host); // gives back list of function nodes that might contain the dynamics

    let _line_span = get_line_span(core_id[0], graph); // get's the line span of function id

    //println!("\n{:?}", line_span);

    //println!("function_core_id: {:?}", core_id[0].clone());
    //println!("module_id: {:?}\n", module_id.clone());
    // 4.5 now to check if of those expressions, if they are arithmetric in nature

    // 5. pass id to subgrapg2_core_dyn to get core dynamics
    let (core_dynamics_ast, _metadata_map_ast) = subgraph2_core_dyn_ast(core_id[0], host).unwrap();

    //println!("\ncore_dynamics_ast[0]: {:?}", core_dynamics_ast[0].clone());

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
pub fn find_pn_dynamics(module_id: i64, host: &str) -> Vec<i64> {
    let graph = subgraph2petgraph(module_id, host);
    // 1. find each function node
    let mut function_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Function"] {
            function_nodes.push(node);
        }
    }
    // 2. check and make sure only expressions in function
    // 3. check number of expressions and decide off that
    let mut functions = Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..function_nodes.len() {
        // grab the subgraph of the given expression
        functions.push(subgraph2petgraph(graph[function_nodes[i]].id, host));
    }
    // get a sense of the number of expressions in each function
    let mut func_counter = 0;
    let mut core_func = Vec::<usize>::new();
    for func in functions.clone() {
        let mut expression_counter = 0;
        let mut primitive_counter = 0;
        for node in func.node_indices() {
            if func[node].labels == ["Expression"] {
                expression_counter += 1;
            }
            if func[node].labels == ["Primitive"] {
                if func[node].properties["name"].to_string() == *"'ast.Mult'" {
                    primitive_counter += 1;
                } else if func[node].properties["name"].to_string() == *"'ast.Add'" {
                    primitive_counter += 1;
                } else if func[node].properties["name"].to_string() == *"'ast.Sub'" {
                    primitive_counter += 1;
                } else if func[node].properties["name"].to_string() == *"'ast.USub'" {
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
        for node in functions[(*c_func)].clone().node_indices() {
            if functions[(*c_func)][node].labels == ["Function"] {
                core_id.push(functions[(*c_func)][node].id);
            }
        }
    }

    core_id
}

pub fn subgrapg2_core_dyn_MET_ast(
    root_node_id: i64,
    host: &str,
) -> Result<(Vec<FirstOrderODE>, HashMap<String, rsmgclient::Node>), MgError> {
    // get the petgraph of the subgraph
    let graph = subgraph2petgraph(root_node_id, host);

    /* MAKE THIS A FUNCTION THAT TAKES IN A PETGRAPH */
    // create the metadata rust rep
    // this will be a map of the name of the node and the metadata node it's attached to with the mapping to our standard metadata struct
    // grab metadata nodes
    let mut metadata_map = HashMap::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Metadata"] {
            for neighbor_node in graph.neighbors_directed(node, Incoming) {
                // NOTE: these names have slightly off formating, the key is: "'name'"
                metadata_map.insert(
                    graph[neighbor_node].properties["name"].to_string().clone(),
                    graph[node].clone(),
                );
            }
        }
    }

    // find all the expressions
    let mut expression_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Expression"] {
            expression_nodes.push(node);
            // println!("Expression Nodes: {:?}", graph[node].clone().id);
        }
    }

    // initialize vector to collect all expressions
    let mut expressions = Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the subgraph of the given expression
        expressions.push(subgraph2petgraph(graph[expression_nodes[i]].id, host));
    }

    // initialize vector to collect all expression wiring graphs
    let mut expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the wiring subgraph of the given expression
        expressions_wiring.push(subgraph_wiring(graph[expression_nodes[i]].id, host).unwrap());
    }

    // now to trim off the un-named filler nodes and filler expressions
    let mut trimmed_expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expressions_wiring.clone().len() {
        let (nodes1, _edges1) = expressions_wiring[i].clone().into_nodes_edges();
        if nodes1.len() > 3 {
            trimmed_expressions_wiring.push(trim_un_named(expressions_wiring[i].clone()));
        }
    }

    // this is the actual convertion
    let mut core_dynamics = Vec::<FirstOrderODE>::new();

    for expr in trimmed_expressions_wiring.clone() {
        let mut root_node = Vec::<NodeIndex>::new();
        for node_index in expr.clone().node_indices() {
            if expr[node_index].labels == ["Opo"] {
                root_node.push(node_index);
            }
        }
        if root_node.len() >= 2 {
            // println!("More than one Opo! Skipping Expression!");
        } else {
            core_dynamics.push(tree_2_MET_ast(expr.clone(), root_node[0]).unwrap());
        }
    }

    Ok((core_dynamics, metadata_map))
}

pub fn subgraph2_core_dyn_ast(
    root_node_id: i64,
    host: &str,
) -> Result<(Vec<Vec<MathExpression>>, HashMap<String, rsmgclient::Node>), MgError> {
    // get the petgraph of the subgraph
    let graph = subgraph2petgraph(root_node_id, host);

    /* MAKE THIS A FUNCTION THAT TAKES IN A PETGRAPH */
    // create the metadata rust rep
    // this will be a map of the name of the node and the metadata node it's attached to with the mapping to our standard metadata struct
    // grab metadata nodes
    let mut metadata_map = HashMap::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Metadata"] {
            for neighbor_node in graph.neighbors_directed(node, Incoming) {
                // NOTE: these names have slightly off formating, the key is: "'name'"
                metadata_map.insert(
                    graph[neighbor_node].properties["name"].to_string().clone(),
                    graph[node].clone(),
                );
            }
        }
    }

    // find all the expressions
    let mut expression_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].labels == ["Expression"] {
            expression_nodes.push(node);
            // println!("Expression Nodes: {:?}", graph[node].clone().id);
        }
    }

    // initialize vector to collect all expressions
    let mut expressions = Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the subgraph of the given expression
        /*println!(
            "These are the nodes for expressions: {:?}",
            graph[expression_nodes[i]].id.clone()
        );*/
        expressions.push(subgraph2petgraph(graph[expression_nodes[i]].id, host));
    }

    // initialize vector to collect all expression wiring graphs
    let mut expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the wiring subgraph of the given expression
        expressions_wiring.push(subgraph_wiring(graph[expression_nodes[i]].id, host).unwrap());
    }

    // now to trim off the un-named filler nodes and filler expressions
    let mut trimmed_expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expressions_wiring.clone().len() {
        let (nodes1, _edges1) = expressions_wiring[i].clone().into_nodes_edges();
        if nodes1.len() > 3 {
            //println!("\n{:?}\n", nodes1.clone());
            // SINCE THE POF'S ARE THE SOURCE OF THE STATE VARIABLES, NOT THE OPI'S. THEY'RE NOT BEING WIRED IN PROPERLY
            trimmed_expressions_wiring.push(trim_un_named(expressions_wiring[i].clone()));
        }
    }

    // this is the actual convertion
    let mut core_dynamics = Vec::<Vec<MathExpression>>::new();

    for expr in trimmed_expressions_wiring.clone() {
        let mut root_node = Vec::<NodeIndex>::new();
        for node_index in expr.clone().node_indices() {
            if expr[node_index].labels == ["Opo"] {
                root_node.push(node_index);
            }
        }
        if root_node.len() >= 2 {
            // println!("More than one Opo! Skipping Expression!");
        } else {
            core_dynamics.push(tree_2_ast(expr.clone(), root_node[0]).unwrap());
        }
    }

    Ok((core_dynamics, metadata_map))
}

fn tree_2_MET_ast(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Result<FirstOrderODE, MgError> {
    let mut fo_eq_vec = Vec::<FirstOrderODE>::new();
    let mut math_vec = Vec::<MathExpressionTree>::new();
    let mut lhs = Vec::<Ci>::new();
    if graph[root_node].labels == ["Opo"] {
        // we first construct the derivative of the first node
        let deriv_name: &str = &graph[root_node].properties["name"].to_string();
        // this will let us know if additional trimming is needed to handle the code implementation of the equations
        let mut step_impl = false;
        // This is very bespoke right now
        // this check is for if it's leibniz notation or not, will need to expand as more cases are creating,
        // currently we convert to leibniz form
        if deriv_name[1..2].to_string() == "d" {
            let deriv = Ci {
                r#type: Some(Function),
                content: Box::new(MathExpression::Mi(Mi(deriv_name[2..3].to_string()))),
            };
            lhs.push(deriv);
        } else {
            step_impl = true;
            let deriv = Ci {
                r#type: Some(Function),
                content: Box::new(MathExpression::Mi(Mi(deriv_name[1..2].to_string()))),
            };
            lhs.push(deriv);
        }
        for node in graph.neighbors_directed(root_node, Outgoing) {
            if graph[node].labels == ["Primitive"] {
                let operate = get_operator_MET(graph.clone(), node); // output -> Operator
                let rhs_arg = get_args_MET(graph.clone(), node); // output -> Vec<MathExpressionTree>
                let mut rhs = MathExpressionTree::Cons(operate, rhs_arg); // MathExpressionTree
                let rhs_flat = flatten_mults(rhs.clone());
                let fo_eq = FirstOrderODE {
                    lhs_var: lhs[0].clone(),
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

pub fn get_args_MET(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Vec<MathExpressionTree> {
    let mut args = Vec::<MathExpressionTree>::new();
    let mut arg_order = Vec::<i64>::new();
    let mut ordered_args = Vec::<MathExpressionTree>::new();

    // idea construct a vector of edge labels for the arguments
    // construct vector of math expressions
    // reorganize based on edge labels vector

    // construct vecs
    for node in graph.neighbors_directed(root_node, Outgoing) {
        // first need to check for operator
        if graph[node].labels == ["Primitive"] {
            let operate = get_operator_MET(graph.clone(), node); // output -> Operator
            let rhs_arg = get_args_MET(graph.clone(), node); // output -> Vec<MathExpressionTree>
            let rhs = MathExpressionTree::Cons(operate, rhs_arg); // MathExpressionTree
            args.push(rhs.clone());
        } else {
            // asummption it is atomic
            let temp_string = graph[node].properties["name"].to_string().clone();
            let arg2 = MathExpressionTree::Atom(MathExpression::Mi(Mi(graph[node].properties
                ["name"]
                .to_string()[1..(temp_string.len() - 1 as usize)]
                .to_string())));
            args.push(arg2.clone());
        }
        // construct order of args
        if let rsmgclient::Value::Int(x) = graph
            .edge_weight(graph.find_edge(root_node, node).unwrap())
            .unwrap()
            .clone()
            .properties["index"]
        {
            arg_order.push(x);
        }
    }

    // fix order of args
    ordered_args = args.clone();
    for (i, ind) in arg_order.iter().enumerate() {
        // the ind'th element of order_args is the ith element of the unordered args
        let temp = std::mem::replace(&mut ordered_args[*ind as usize], args[i as usize].clone());
    }

    ordered_args
}

// this gets the operator from the node name
pub fn get_operator_MET(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Operator {
    let mut op = Vec::<Operator>::new();
    if graph[root_node].properties["name"].to_string() == *"'ast.Mult'" {
        op.push(Operator::Multiply);
    } else if graph[root_node].properties["name"].to_string() == *"'ast.Add'" {
        op.push(Operator::Add);
    } else if graph[root_node].properties["name"].to_string() == *"'ast.Sub'" {
        op.push(Operator::Subtract);
    } else if graph[root_node].properties["name"].to_string() == *"'ast.USub'" {
        op.push(Operator::Subtract);
    } else if graph[root_node].properties["name"].to_string() == *"'ast.Div'" {
        op.push(Operator::Divide);
    } else {
        op.push(Operator::Other(
            graph[root_node].properties["name"].to_string(),
        ));
    }
    op[0].clone()
}

fn tree_2_ast(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Result<Vec<MathExpression>, MgError> {
    let mut math_vec = Vec::<MathExpression>::new();

    if graph[root_node].labels == ["Opo"] {
        // we first construct the derivative of the first node
        let deriv_name: &str = &graph[root_node].properties["name"].to_string();
        // this will let us know if additional trimming is needed to handle the code implementation of the equations
        let mut step_impl = false;
        // This is very bespoke right now
        // this check is for if it's leibniz notation or not, will need to expand as more cases are creating,
        // currently we convert to leibniz form
        if deriv_name[1..2].to_string() == "d" {
            let deriv = MathExpression::Mfrac(
                Box::new(MathExpression::Mrow(Mrow(vec![
                    MathExpression::Mi(Mi(deriv_name[1..2].to_string())),
                    MathExpression::Mi(Mi(deriv_name[2..3].to_string())),
                ]))),
                Box::new(MathExpression::Mrow(Mrow(vec![
                    MathExpression::Mi(Mi(deriv_name[3..4].to_string())),
                    MathExpression::Mi(Mi(deriv_name[4..5].to_string())),
                ]))),
            );
            math_vec.push(deriv);
        } else {
            step_impl = true;
            let deriv = MathExpression::Mfrac(
                Box::new(MathExpression::Mrow(Mrow(vec![
                    MathExpression::Mi(Mi("d".to_string())),
                    MathExpression::Mi(Mi(deriv_name[1..2].to_string())),
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
            if graph[node].labels == ["Primitive"]
                && graph[node].properties["name"].to_string() != *"'ast.USub'"
            {
                first_op.push(get_operator(graph.clone(), node));
                let mut arg1 = get_args(graph.clone(), node);
                if graph[node].properties["name"].to_string() == *"'ast.Mult'" {
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
                        if *ent == MathExpression::Mo(Operator::Other("'ast.USub'".to_string())) {
                            usub_exist0 = true;
                            usub_idx.push(i.try_into().unwrap());
                        }
                    }
                    for (i, ent) in arg1[1].clone().iter().enumerate() {
                        if *ent == MathExpression::Mo(Operator::Other("'ast.USub'".to_string())) {
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
            let ref_name = deriv_name[1..2].to_string();
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

    //println!("Not reversed mathml: {:?}", math_vec.clone());

    // do to how the expression tree in generated we need to reverse the order
    let mut reversed_final_math = Vec::<MathExpression>::new();
    let vec_len_temp = math_vec.clone().len();

    reversed_final_math.extend_from_slice(&math_vec.clone()[0..2]);

    for (i, j) in math_vec.clone().iter().rev().enumerate() {
        if i != vec_len_temp && i != (vec_len_temp - 1) && i != (vec_len_temp - 2) {
            reversed_final_math.push(j.clone());
        }
    }

    //println!("reversed mathml: {:?}", reversed_final_math.clone());

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
    let mut arg2_term_ind = Vec::<i32>::new();

    arg2_term_ind = terms_indicies(arg2.clone());

    // check if need to swap operator signs
    /* Is this running properly, not removing Mo(Other("'USub'")) in I equation in SIR */
    if arg1[0] == Mo(Operator::Other("'USub'".to_string())) {
        println!("USub dist happens"); // This is never running
        println!("Entry 1"); // operator starts at begining of arg2
        if arg2_term_ind[0] == 0 {
            for (i, ind) in arg2_term_ind.iter().enumerate() {
                if arg2[(*ind as usize)] == Mo(Operator::Add) {
                    arg2[(*ind as usize)] = Mo(Operator::Subtract);
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    }
                } else {
                    arg2[(*ind as usize)] = Mo(Operator::Add);
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    }
                }
            }
        } else {
            println!("Entry 2");
            // operator doesn't start at beginning so have to add it manually
            arg_dist.push(Mo(Operator::Subtract));
            arg_dist.extend_from_slice(&arg2.clone()[0..(arg2_term_ind[0] - 1) as usize]);
            //arg_dist.push(Mo(Operator::Multiply));
            let vec_len1 = arg1.clone().len() - 1;
            arg_dist.extend_from_slice(&arg1[1..vec_len1]);
            for (i, ind) in arg2_term_ind.iter().enumerate() {
                if arg2[(*ind as usize)] == Mo(Operator::Add) {
                    arg2[(*ind as usize)] = Mo(Operator::Subtract);
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        let vec_len1 = arg1.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg1.clone()[1..vec_len1]);
                    }
                } else {
                    arg2[(*ind as usize)] = Mo(Operator::Add);
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()[1..]);
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
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
                if arg2[(*ind as usize)] == Mo(Operator::Add) {
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    }
                } else if (i + 1) != arg2_term_ind.len() {
                    arg_dist.extend_from_slice(
                        &arg2.clone()[(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                    );
                    //arg_dist.push(Mo(Operator::Multiply));
                    arg_dist.extend_from_slice(&arg1.clone()); // check
                } else {
                    // last of the expression case
                    let vec_len = arg2.clone().len() - 1;
                    arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                    //arg_dist.push(Mo(Operator::Multiply));
                    arg_dist.extend_from_slice(&arg1.clone()); // check
                }
            }
        } else {
            println!("Entry 4");
            // don't swap operators manual beginning push
            arg_dist.extend_from_slice(&arg2.clone()[0..(arg2_term_ind[0] - 1) as usize]);
            //arg_dist.push(Mo(Operator::Multiply));
            let vec_len1 = arg1.clone().len(); // let vec_len1 = arg1.clone().len() - 1;
            arg_dist.extend_from_slice(&arg1[1..vec_len1]);
            for (i, ind) in arg2_term_ind.iter().enumerate() {
                if arg2[*ind as usize] == Mo(Operator::Add) {
                    if (i + 1) != arg2_term_ind.len() {
                        arg_dist.extend_from_slice(
                            &arg2.clone()
                                [(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                        );
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    } else {
                        // last of the expression case
                        let vec_len = arg2.clone().len() - 1;
                        arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                        //arg_dist.push(Mo(Operator::Multiply));
                        arg_dist.extend_from_slice(&arg1.clone()); // check
                    }
                } else if (i + 1) != arg2_term_ind.len() {
                    arg_dist.extend_from_slice(
                        &arg2.clone()[(*ind as usize)..(arg2_term_ind.clone()[i + 1] - 1) as usize],
                    );
                    //arg_dist.push(Mo(Operator::Multiply));
                    arg_dist.extend_from_slice(&arg1.clone()); // check
                } else {
                    // last of the expression case
                    let vec_len = arg2.clone().len() - 1;
                    arg_dist.extend_from_slice(&arg2.clone()[(*ind as usize)..vec_len]);
                    //arg_dist.push(Mo(Operator::Multiply));
                    arg_dist.extend_from_slice(&arg1.clone()); // check
                }
            }
        }
    }
    arg_dist
}

// this will get the arguments for an operator/primitive of the graph, assumes it's binary
fn get_args(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Vec<Vec<MathExpression>> {
    let mut op = Vec::<MathExpression>::new();
    // need to construct the vector of length 2 with temporary filling since we might get the second
    // element first and tuples can't be dynamically indexed in rust
    let temp_op = Vec::<MathExpression>::new();
    let mut args = vec![temp_op; 2];

    for (i, node) in graph.neighbors_directed(root_node, Outgoing).enumerate() {
        if graph[node].labels == ["Primitive"]
            && graph[node].properties["name"].to_string() == *"'USub'"
        {
            op.push(get_operator(graph.clone(), node));
            for node1 in graph.neighbors_directed(node, Outgoing) {
                let temp_mi = MathExpression::Mi(Mi(graph.clone()[node1].properties["name"]
                    .to_string()
                    [1..(graph.clone()[node1].properties["name"].to_string().len() - 1_usize)]
                    .to_string()));
                args[i].push(op[0].clone());
                args[i].push(temp_mi.clone());
            }
        } else if graph[node].labels == ["Opi"] || graph[node].labels == ["Literal"] {
            let temp_mi = MathExpression::Mi(Mi(graph.clone()[node].properties["name"]
                .to_string()
                [1..(graph.clone()[node].properties["name"].to_string().len() - 1_usize)]
                .to_string()));
            args[i].push(temp_mi.clone());
        } else {
            let n_args = get_args(graph.clone(), node);
            let mut temp_vec = Vec::<MathExpression>::new();
            temp_vec.extend_from_slice(&n_args[0]);
            temp_vec.push(get_operator(graph.clone(), node));
            temp_vec.extend_from_slice(&n_args[1]);
            args[i].extend_from_slice(&temp_vec.clone());
        }
    }

    args
}

// this gets the operator from the node name
fn get_operator(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> MathExpression {
    let mut op = Vec::<MathExpression>::new();
    if graph[root_node].properties["name"].to_string() == *"'ast.Mult'" {
        op.push(Mo(Operator::Multiply));
    } else if graph[root_node].properties["name"].to_string() == *"'ast.Add'" {
        op.push(Mo(Operator::Add));
    } else if graph[root_node].properties["name"].to_string() == *"'ast.Sub'" {
        op.push(Mo(Operator::Subtract));
    } else if graph[root_node].properties["name"].to_string() == *"'ast.Div'" {
        op.push(Mo(Operator::Divide));
    } else {
        op.push(Mo(Operator::Other(
            graph[root_node].properties["name"].to_string(),
        )));
    }
    op[0].clone()
}

// this currently only works for un-named nodes that are not chained or have multiple incoming/outgoing edges
fn trim_un_named(
    graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
) -> petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> {
    // first create a cloned version of the graph we can modify while iterating over it.
    let mut bypass_graph = graph.clone();

    // iterate over the graph and add a new edge to bypass the un-named nodes
    for node_index in graph.node_indices() {
        if graph[node_index].properties["name"].to_string() == *"'un-named'" {
            let mut bypass = Vec::<NodeIndex>::new();
            for node1 in graph.neighbors_directed(node_index, Incoming) {
                bypass.push(node1);
            }
            for node2 in graph.neighbors_directed(node_index, Outgoing) {
                bypass.push(node2);
            }
            // one incoming one outgoing
            if bypass.len() == 2 {
                bypass_graph.add_edge(
                    bypass[0],
                    bypass[1],
                    graph
                        .edge_weight(graph.find_edge(bypass[0], node_index).unwrap())
                        .unwrap()
                        .clone(),
                );
            } else if bypass.len() > 2 {
                // this operates on the assumption that there maybe multiple references to the port
                // (incoming arrows) but only one outgoing arrow, this seems to be the case based on
                // data too.

                let end_node_idx = bypass.len() - 1;
                for (i, _ent) in bypass[0..end_node_idx].iter().enumerate() {
                    // this iterates over all but the last entry in the bypass vec
                    bypass_graph.add_edge(
                        bypass[i],
                        bypass[end_node_idx],
                        graph
                            .edge_weight(graph.find_edge(bypass[i], node_index).unwrap())
                            .unwrap()
                            .clone(),
                    );
                }
            }
        }
    }

    // now we perform a filter_map to remove the un-named nodes and only the bypass edge will remain to connect the nodes
    // we also remove the unpack node if it is present here as well

    bypass_graph.filter_map(
        |node_index, _edge_index| {
            if !(bypass_graph[node_index].properties["name"].to_string() == *"'un-named'"
                || bypass_graph[node_index].properties["name"].to_string() == *"'unpack'")
            {
                Some(graph[node_index].clone())
            } else {
                None
            }
        },
        |_node_index, edge_index| Some(edge_index.clone()),
    )
}

fn subgraph_wiring(
    module_id: i64,
    host: &str,
) -> Result<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>, MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some(host.to_string()),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // create query for nodes
    let query1 = format!(
        "MATCH (n)-[*]->(m) WHERE id(n) = {}
        MATCH q = (l)<-[r:Wire]-(m)
        WITH reduce(output = [], m IN nodes(q) | output + m ) AS nodes1
        UNWIND nodes1 AS nodes2
        WITH DISTINCT nodes2
        return collect(nodes2)",
        module_id
    );

    // Run Query for nodes
    connection.execute(&query1, None)?;

    // collect nodes into list
    let mut node_list = Vec::<Node>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        node_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::Node(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    // create query for edges
    let query2 = format!(
        "MATCH (n)-[*]->(m) WHERE id(n) = {}
        MATCH q = (l)<-[r:Wire]-(m)
        WITH reduce(output = [], m IN relationships(q) | output + m ) AS edges1
        UNWIND edges1 AS edges2
        WITH DISTINCT edges2
        return collect(edges2)",
        module_id
    );

    // Run Query for edges
    connection.execute(&query2, None)?;

    // collect edges into list
    let mut edge_list = Vec::<Relationship>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        edge_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::Relationship(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    // Create a petgraph graph
    let mut graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> = Graph::new();

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
        for node_idx in nodes.clone() {
            if graph[node_idx].id == edge.start_id {
                src.push(node_idx);
            }
            if graph[node_idx].id == edge.end_id {
                tgt.push(node_idx);
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0], tgt[0], edge);
        }
    }
    Ok(graph)
}

fn subgraph2petgraph(
    module_id: i64,
    host: &str,
) -> petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> {
    let (x, y) = get_subgraph(module_id, host).unwrap();

    // Create a petgraph graph
    let mut graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> = Graph::new();

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
        for node_idx in nodes.clone() {
            if graph[node_idx].id == edge.start_id {
                src.push(node_idx);
            }
            if graph[node_idx].id == edge.end_id {
                tgt.push(node_idx);
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0], tgt[0], edge);
        }
    }
    graph
}

pub fn get_subgraph(module_id: i64, host: &str) -> Result<(Vec<Node>, Vec<Relationship>), MgError> {
    // construct the query that will delete the module with a given unique identifier

    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some(host.to_string()),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // create query for nodes
    let query1 = format!(
        "MATCH p = (n)-[r*]->(m) WHERE id(n) = {}
    WITH reduce(output = [], n IN nodes(p) | output + n ) AS nodes1
    UNWIND nodes1 AS nodes2
    WITH DISTINCT nodes2
    return collect(nodes2)",
        module_id
    );

    // Run Query for nodes
    connection.execute(&query1, None)?;

    // collect nodes into list
    let mut node_list = Vec::<Node>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        node_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::Node(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    // create query for edges
    let query2 = format!(
        "MATCH p = (n)-[r*]->(m) WHERE id(n) ={}
        WITH reduce(output = [], n IN relationships(p) | output + n ) AS edges1
        UNWIND edges1 AS edges2
        WITH DISTINCT edges2
        return collect(edges2)",
        module_id
    );

    // Run Query for edges
    connection.execute(&query2, None)?;

    // collect edges into list
    let mut edge_list = Vec::<Relationship>::new();
    if let Value::List(xs) = &connection.fetchall()?[0].values[0] {
        edge_list = xs
            .iter()
            .filter_map(|x| match x {
                Value::Relationship(x) => Some(x.clone()),
                _ => None,
            })
            .collect();
    }
    connection.commit()?;

    Ok((node_list, edge_list))
}
