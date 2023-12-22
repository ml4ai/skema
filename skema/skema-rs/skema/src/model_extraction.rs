use crate::config::Config;
use crate::ValueL;

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

/// This struct is the node struct for the constructed petgraph
#[derive(Clone, Debug)]
pub struct ModelNode {
    id: i64,
    label: String,
    name: Option<String>,
    value: Option<ValueL>,
}

/// This struct is the edge struct for the constructed petgraph
#[derive(Clone, Debug)]
pub struct ModelEdge {
    id: i64,
    src_id: i64,
    tgt_id: i64,
    index: Option<i64>,
    refer: Option<i64>,
}

/**
 * This is the main function call for model extraction.
 *
 * Parameters:
 * - module_id: i64 -> This is the top level id of the gromet module in memgraph.
 * - config: Config -> This is a config struct for connecting to memgraph
 *
 * Returns:
 * - Vector of FirstOrderODE -> This vector of structs is used to construct a PetriNet or RegNet further down the pipeline
 *
 * Assumptions:
 * - As of right now, we can always assume the code has been sliced to only one relevant function which contains the
 * core dynamics in it somewhere
 *
 * Notes:
 * - FirstOrderODE is primarily composed of a LHS and a RHS,
 *      - LHS is just a Mi object of the state being differentiated. There are additional fields for the LHS but only the
 *      content field is used in downstream inference for now.
 *      - RHS is where the bulk of the inference happens, it produces an expression tree, hence the MET -> Math Expression Tree.
 *          Every operator has a vector of arguments. (order matters)
 */
#[allow(non_snake_case)]
pub async fn module_id2mathml_MET_ast(module_id: i64, config: Config) -> Vec<FirstOrderODE> {
    let mut core_dynamics_ast = Vec::<FirstOrderODE>::new();

    let core_id = find_pn_dynamics(module_id, config.clone()).await;

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

/**
 * This function finds the core dynamics and returns a vector of node id's that meet the criteria for identification
 *
 * Based on the fact we are getting in only the function we expect to have dynamics, this should just be depricated in the future
 * and replaced with a inference to move from the module_id to the top level function node id, however for now we should keep it
 * as a simple heuristic because it is used in the original code2amr (zip repo) endpoint which would need to be updated first.
 *
 * Plus the case when it fails defaults to a emptry AMR which is preferable to crashing.
*/
#[allow(clippy::if_same_then_else)]
pub async fn find_pn_dynamics(module_id: i64, config: Config) -> Vec<i64> {
    let graph = subgraph2petgraph(module_id, config.clone()).await;
    // 1. find each function node
    let mut function_nodes = Vec::<NodeIndex>::new();
    for node in graph.node_indices() {
        if graph[node].label == *"Function" {
            function_nodes.push(node);
        }
    }
    // 2. check and make sure only expressions in function
    // 3. check number of expressions and decide off that
    let mut functions = Vec::<petgraph::Graph<ModelNode, ModelEdge>>::new();
    for i in 0..function_nodes.len() {
        // grab the subgraph of the given expression
        functions.push(subgraph2petgraph(graph[function_nodes[i]].id, config.clone()).await);
    }
    // get a sense of the number of expressions in each function
    let mut func_counter = 0;
    let mut core_func = Vec::<usize>::new();
    for func in &functions {
        let mut expression_counter = 0;
        let mut primitive_counter = 0;
        for node in func.node_indices() {
            if func[node].label == *"Expression" {
                expression_counter += 1;
            }
            if func[node].label == *"Primitive" {
                if *func[node].name.as_ref().unwrap() == "ast.Mult".to_string() {
                    primitive_counter += 1;
                } else if *func[node].name.as_ref().unwrap() == "ast.Add".to_string() {
                    primitive_counter += 1;
                } else if *func[node].name.as_ref().unwrap() == "ast.Sub".to_string() {
                    primitive_counter += 1;
                } else if *func[node].name.as_ref().unwrap() == "ast.USub".to_string() {
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
            if functions[*c_func][node].label == *"Function" {
                core_id.push(functions[*c_func][node].id);
            }
        }
    }

    core_id
}

/**
 * Once the function node has been identified, this function takes it from there to extract the vector of FirstOrderODE's
 *
 * This is based heavily on the assumption that each equation is in a seperate expression which breaks for the vector case.
 */
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
        if graph[node].label == *"Expression" {
            expression_nodes.push(node);
        }
    }

    let mut core_dynamics = Vec::<FirstOrderODE>::new();

    // initialize vector to collect all expression wiring graphs
    for i in 0..expression_nodes.len() {
        // grab the wiring subgraph of the given expression
        let mut sub_w = subgraph_wiring(graph[expression_nodes[i]].id, config.clone())
            .await
            .unwrap();
        let mut prim_counter = 0;
        let mut has_call = false;
        for node_index in sub_w.node_indices() {
            if sub_w[node_index].label == *"Primitive" {
                prim_counter += 1;
                if *sub_w[node_index].name.as_ref().unwrap() == "_call" {
                    has_call = true;
                }
            }
        }
        if sub_w.node_count() > 3 && !(prim_counter == 1 && has_call) && prim_counter != 0 {
            println!("--------------------");
            println!("expression: {}", graph[expression_nodes[i]].id);
            // the call expressions get referenced by multiple top level expressions, so deleting the nodes in it breaks the other graphs. Need to pass clone of expression subgraph so references to original has all the nodes.
            if has_call {
                sub_w = trim_calls(sub_w.clone())
            }
            let expr = trim_un_named(&mut sub_w);
            let mut root_node = Vec::<NodeIndex>::new();
            for node_index in expr.node_indices() {
                if expr[node_index].label.clone() == *"Opo" {
                    root_node.push(node_index);
                }
            }
            if root_node.len() >= 2 {
                println!("More than one Opo! Skipping Expression!");
            } else {
                core_dynamics.push(tree_2_MET_ast(expr, root_node[0]).unwrap());
            }
        }
    }

    Ok(core_dynamics)
}

/**
 * This function is designed to take in a petgraph instance of a wires only expression subgraph and output a FirstOrderODE equations representing it.
 */
#[allow(non_snake_case)]
fn tree_2_MET_ast(
    graph: &mut petgraph::Graph<ModelNode, ModelEdge>,
    root_node: NodeIndex,
) -> Result<FirstOrderODE, Error> {
    let mut fo_eq_vec = Vec::<FirstOrderODE>::new();
    let _math_vec = Vec::<MathExpressionTree>::new();
    let mut lhs = Vec::<Ci>::new();
    if graph[root_node].label == *"Opo" {
        // we first construct the derivative of the first node
        let deriv_name: &str = graph[root_node].name.as_ref().unwrap();
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
            if graph[node].label.clone() == *"Primitive" {
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

/// This is a recursive function that walks along the wired subgraph of an expression to construct the expression tree
#[allow(non_snake_case)]
pub fn get_args_MET(
    graph: &petgraph::Graph<ModelNode, ModelEdge>,
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
        if graph[node].label.clone() == *"Primitive" {
            let operate = get_operator_MET(graph, node); // output -> Operator
            let rhs_arg = get_args_MET(graph, node); // output -> Vec<MathExpressionTree>
            let rhs = MathExpressionTree::Cons(operate, rhs_arg); // MathExpressionTree
            args.push(rhs.clone());
        } else {
            // asummption it is atomic
            if graph[node].label.clone() == *"Literal" {
                let temp_string = graph[node].value.clone().unwrap().value.replace('\"', "");
                let arg2 = MathExpressionTree::Atom(MathExpression::Mi(Mi(temp_string.clone())));
                args.push(arg2.clone());
            } else {
                let temp_string = graph[node].name.as_ref().unwrap().clone();
                let arg2 = MathExpressionTree::Atom(MathExpression::Mi(Mi(temp_string.clone())));
                args.push(arg2.clone());
            }
        }

        // construct order of args
        let x = graph
            .edge_weight(graph.find_edge(root_node, node).unwrap())
            .unwrap()
            .index
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

/// This gets the operator from the node name
#[allow(non_snake_case)]
#[allow(clippy::if_same_then_else)]
pub fn get_operator_MET(
    graph: &petgraph::Graph<ModelNode, ModelEdge>,
    root_node: NodeIndex,
) -> Operator {
    let mut op = Vec::<Operator>::new();
    if *graph[root_node].name.as_ref().unwrap() == "ast.Mult".to_string() {
        op.push(Operator::Multiply);
    } else if *graph[root_node].name.as_ref().unwrap() == "ast.Add" {
        op.push(Operator::Add);
    } else if *graph[root_node].name.as_ref().unwrap() == "ast.Sub" {
        op.push(Operator::Subtract);
    } else if *graph[root_node].name.as_ref().unwrap() == "ast.USub" {
        op.push(Operator::Subtract);
    } else if *graph[root_node].name.as_ref().unwrap() == "ast.Div" {
        op.push(Operator::Divide);
    } else {
        op.push(Operator::Other(graph[root_node].name.clone().unwrap()));
    }
    op[0].clone()
}

/**
 * This function takes in a wiring only petgraph of an expression and trims off the un-named nodes and unpack nodes.
 *
 * This is done by creating new edges that bypass the un-named nodes and then deleting them from the graph.
 * For deleting the unpacks, the assumption is they are always terminal in the subgraph and can be deleted freely.
 *
 * Concerns:
 * - I don't think this will work if there are multiple un-named nodes changed together. I haven't seen this in practice,
 * but I think it's possible. So something to keep in mind.
 */
fn trim_un_named(
    graph: &mut petgraph::Graph<ModelNode, ModelEdge>,
) -> &mut petgraph::Graph<ModelNode, ModelEdge> {
    // first create a cloned version of the graph we can modify while iterating over it.

    // iterate over the graph and add a new edge to bypass the un-named nodes
    for node_index in graph.node_indices() {
        if graph[node_index].clone().name.unwrap().clone() == *"un-named" {
            let mut bypass = Vec::<NodeIndex>::new();
            let mut outgoing_bypass = Vec::<NodeIndex>::new();
            for node1 in graph.neighbors_directed(node_index, Incoming) {
                bypass.push(node1);
            }
            for node2 in graph.neighbors_directed(node_index, Outgoing) {
                outgoing_bypass.push(node2);
            }
            // one incoming one outgoing
            if bypass.len() == 1 && outgoing_bypass.len() == 1 {
                // annoyingly have to pull the edge/Relation to insert into graph
                graph.add_edge(
                    bypass[0],
                    outgoing_bypass[0],
                    graph
                        .edge_weight(graph.find_edge(bypass[0], node_index).unwrap())
                        .unwrap()
                        .clone(),
                );
            } else if bypass.len() >= 2 && outgoing_bypass.len() == 1 {
                // this operates on the assumption that there maybe multiple references to the port
                // (incoming arrows) but only one outgoing arrow, this seems to be the case based on
                // data too.

                for (i, _ent) in bypass.iter().enumerate() {
                    // this iterates over all but the last entry in the bypass vec
                    graph.add_edge(
                        bypass[i],
                        outgoing_bypass[0],
                        graph
                            .edge_weight(graph.find_edge(bypass[i], node_index).unwrap())
                            .unwrap()
                            .clone(),
                    );
                }
            }
        }
    }

    // now we remove the un-named nodes and only the bypass edge will remain to connect the nodes
    // we also remove the unpack node if it is present here as well
    for node_index in graph.node_indices().rev() {
        if graph[node_index].name.clone().unwrap() == *"un-named"
            || graph[node_index].name.clone().unwrap() == *"unpack"
        {
            graph.remove_node(node_index);
        }
    }

    graph
}

/// This function takes in a node id (typically that of an expression subgraph) and returns a
/// petgraph subgraph of only the wire type edges
async fn subgraph_wiring(
    module_id: i64,
    config: Config,
) -> Result<petgraph::Graph<ModelNode, ModelEdge>, Error> {
    let mut node_list = Vec::<ModelNode>::new();
    let mut edge_list = Vec::<ModelEdge>::new();

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
        let modelnode = ModelNode {
            id: node.id(),
            label: node.labels()[0].to_string(),
            name: node.get::<String>("name").ok(),
            value: node.get::<ValueL>("value").ok(),
        };
        node_list.push(modelnode.clone());
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
        let modeledge = ModelEdge {
            id: edge.id(),
            src_id: edge.start_node_id(),
            tgt_id: edge.end_node_id(),
            index: edge.get::<i64>("index").ok(),
            refer: edge.get::<i64>("refer").ok(),
        };
        edge_list.push(modeledge);
    }

    let mut graph: petgraph::Graph<ModelNode, ModelEdge> = Graph::new();

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
            if graph[*node_idx].id == edge.src_id {
                src.push(*node_idx);
            }
            if graph[*node_idx].id == edge.tgt_id {
                tgt.push(*node_idx);
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0], tgt[0], edge);
        }
    }
    Ok(graph)
}

/// This function takes in a node id and returns a petgraph represention of the memgraph graph
async fn subgraph2petgraph(
    module_id: i64,
    config: Config,
) -> petgraph::Graph<ModelNode, ModelEdge> {
    let (x, y) = get_subgraph(module_id, config.clone()).await.unwrap();

    // Create a petgraph graph
    let mut graph: petgraph::Graph<ModelNode, ModelEdge> = Graph::new();

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
            if graph[*node_idx].id == edge.src_id {
                src.push(*node_idx);
            }
            if graph[*node_idx].id == edge.tgt_id {
                tgt.push(*node_idx);
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0], tgt[0], edge);
        }
    }
    graph
}

/// This function takes in a node id and returns the nodes and edges in it
pub async fn get_subgraph(
    module_id: i64,
    config: Config,
) -> Result<(Vec<ModelNode>, Vec<ModelEdge>), Error> {
    let mut node_list = Vec::<ModelNode>::new();
    let mut edge_list = Vec::<ModelEdge>::new();

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
        let modelnode = ModelNode {
            id: node.id(),
            label: node.labels()[0].to_string(),
            name: node.get::<String>("name").ok(),
            value: node.get::<ValueL>("value").ok(),
        };
        node_list.push(modelnode);
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
        let modeledge = ModelEdge {
            id: edge.id(),
            src_id: edge.start_node_id(),
            tgt_id: edge.end_node_id(),
            index: edge.get::<i64>("index").ok(),
            refer: edge.get::<i64>("refer").ok(),
        };
        edge_list.push(modeledge);
    }

    Ok((node_list, edge_list))
}

// this does special trimming to handle function calls
pub fn trim_calls(
    graph: petgraph::Graph<ModelNode, ModelEdge>,
) -> petgraph::Graph<ModelNode, ModelEdge> {
    let mut graph_clone = graph.clone();

    // This will be all the nodes to be deleted
    let mut inner_nodes = Vec::<NodeIndex>::new();
    // find the call nodes
    for node_index in graph.node_indices() {
        if graph[node_index].clone().name.unwrap().clone() == *"_call" {
            // we now trace up the incoming path until we hit a primitive,
            // this will be the start node for the new edge.

            // initialize trackers
            let mut node_start = node_index;
            let mut node_end = node_index;
            let mut i_inner_nodes = Vec::<NodeIndex>::new();

            // find end node and track path
            for node in graph.neighbors_directed(node_index, Outgoing) {
                if graph
                    .edge_weight(graph.find_edge(node_index, node).unwrap())
                    .unwrap()
                    .index
                    .unwrap()
                    == 0
                {
                    let mut temp = to_terminal(graph.clone(), node);
                    node_end = temp.0;
                    i_inner_nodes.append(&mut temp.1);
                }
            }

            // find start primtive node and track path
            for node in graph.neighbors_directed(node_index, Incoming) {
                let mut temp = to_primitive(graph.clone(), node);
                node_start = temp.0;
                i_inner_nodes.append(&mut temp.1);
            }

            // add edge from start to end node, with weight from start node a matching outgoing node form it
            for node in graph.clone().neighbors_directed(node_start, Outgoing) {
                for node_p in i_inner_nodes.iter() {
                    if node == *node_p {
                        graph_clone.add_edge(
                            node_start,
                            node_end,
                            graph
                                .clone()
                                .edge_weight(graph.clone().find_edge(node_start, node).unwrap())
                                .unwrap()
                                .clone(),
                        );
                    }
                }
            }
            // we keep track all the node indexes we found while tracing the path and delete all
            // intermediate nodes.
            i_inner_nodes.push(node_index);
            inner_nodes.append(&mut i_inner_nodes.clone());
        }
    }
    inner_nodes.sort();
    for node in inner_nodes.iter().rev() {
        graph_clone.remove_node(*node);
    }
    graph_clone
}

pub fn to_terminal(
    graph: petgraph::Graph<ModelNode, ModelEdge>,
    node_index: NodeIndex,
) -> (NodeIndex, Vec<NodeIndex>) {
    let mut node_vec = Vec::<NodeIndex>::new();
    let mut end_node = node_index;
    // if there another node deeper
    // else pass original input node out and an empty path vector
    if graph.neighbors_directed(node_index, Outgoing).count() != 0 {
        node_vec.push(node_index); // add current node to path list
        for node in graph.neighbors_directed(node_index, Outgoing) {
            // pass next node forward
            let mut temp = to_terminal(graph.clone(), node);
            end_node = temp.0; // make end_node
            node_vec.append(&mut temp.1); // append previous path nodes
        }
    }
    (end_node, node_vec)
}

// incoming walker to first primitive (NOTE: assumes input is not a primitive)
pub fn to_primitive(
    graph: petgraph::Graph<ModelNode, ModelEdge>,
    node_index: NodeIndex,
) -> (NodeIndex, Vec<NodeIndex>) {
    let mut node_vec = Vec::<NodeIndex>::new();
    let mut end_node = node_index;
    node_vec.push(node_index);
    for node in graph.neighbors_directed(node_index, Incoming) {
        if graph[node].label.clone() != *"Primitive" {
            let mut temp = to_primitive(graph.clone(), node);
            end_node = temp.0;
            node_vec.append(&mut temp.1);
        } else {
            end_node = node;
        }
    }
    (end_node, node_vec)
}
