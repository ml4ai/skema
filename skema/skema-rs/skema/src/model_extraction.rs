use crate::config::Config;
use mathml::ast::operator::Operator;
pub use mathml::mml2pn::{ACSet, Term};
use petgraph::prelude::*;

use std::string::ToString;

// new imports
use mathml::ast::Ci;

use mathml::ast::Type::Function;
use mathml::ast::{MathExpression, Mi};
use mathml::parsers::first_order_ode::{flatten_mults, FirstOrderODE};
use mathml::parsers::math_expression_tree::MathExpressionTree;

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
