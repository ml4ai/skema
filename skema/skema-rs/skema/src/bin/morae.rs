use mathml::ast::Operator;
use mathml::expression::Atom;
use mathml::expression::Expr;
use petgraph::dot::{Config, Dot};
use petgraph::matrix_graph::IndexType;
use petgraph::prelude::*;
use petgraph::*;
use rsmgclient::{ConnectParams, Connection, MgError, Node, Relationship, Value};
use std::collections::HashMap;

fn main() {
    let module_id = 460;
    let graph = subgraph2petgraph(module_id);

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
        }
    }

    // initialize vector to collect all expressions
    let mut expressions = Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the subgraph of the given expression
        expressions.push(subgraph2petgraph(graph[expression_nodes[i]].id.clone()));
    }

    // initialize vector to collect all expression wiring graphs
    let mut expressions_wiring =
        Vec::<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>>::new();
    for i in 0..expression_nodes.len() {
        // grab the wiring subgraph of the given expression
        expressions_wiring.push(subgraph_wiring(graph[expression_nodes[i]].id.clone()).unwrap());
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

    // now we convert the following into a Expr to get converted into a petri net
    // first we have to get the parent node index to pass into the function
    let mut root_node = Vec::<NodeIndex>::new();
    for node_index in graph.clone().node_indices() {
        if graph.clone()[node_index].labels == ["Opo"] {
            root_node.push(node_index);
        }
    }
    if root_node.len() >= 2 {
        panic!("More than one Opo!");
    }

    let expr1 = tree_2_expr(trimmed_expressions_wiring[1], root_node[0]).unwrap();

    println!("{:?}", expr1);

    // debugging outputs
    /*for node_idx in expressions_wiring[1].clone().node_indices() {
        if expressions_wiring[1].clone()[node_idx].properties["name"].to_string()
            == "'un-named'".to_string()
        {
            println!("{:?}", node_idx);
        }
    }*/
    /*for i in 0..trimmed_expressions_wiring.len() {
        println!("{:?}", graph[expression_nodes[i]].id);
        println!(
            "Nodes in wiring subgraph: {}",
            trimmed_expressions_wiring[i].node_count()
        );
        println!(
            "Edges in wiring subgraph: {}",
            trimmed_expressions_wiring[i].edge_count()
        );
    }*/
    //let (nodes1, edges1) = expressions_wiring[1].clone().into_nodes_edges();
    /*for edge in edges1 {
        let source = edge.source();
        let target = edge.target();
        let edge_weight = edge.weight;
        println!(
            "Edge from {:?} to {:?} with weight {:?}",
            source, target, edge_weight
        );
    }*/
    /*for node in nodes1 {
        println!("{:?}", node);
    }*/
    /*println!(
        "{:?}",
        Dot::with_config(&expressions_wiring[1], &[Config::EdgeNoLabel])
    );*/
}

fn tree_2_expr(
    mut graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
    root_node: NodeIndex,
) -> Result<Expr, MgError> {
    // initialize intermediate struct, need to figure out
    let mut op_vec = Vec::<Operator>::new();
    let mut args_vec = Vec::<Expr>::new();
    let mut expr_name = String::from("");

    if graph.clone()[root_node].labels == ["Opo"] {
        // starting node in expression tree, traverse down one node and then start parse
        for node in graph.neighbors_directed(root_node, Outgoing) {
            if graph.clone()[node].labels == ["Primitive"]
                && !graph.clone()[node].properties["name"].to_string() == "'unpack'".to_string()
            {
                // make an operator type based on operation
                if graph.clone()[node].properties["name"].to_string() == "'+'".to_string() {
                    op_vec.push(Operator::Add);
                } else if graph.clone()[node].properties["name"].to_string() == "'-'".to_string() {
                    op_vec.push(Operator::Subtract);
                } else if graph.clone()[node].properties["name"].to_string() == "'*'".to_string() {
                    op_vec.push(Operator::Multiply);
                } else if graph.clone()[node].properties["name"].to_string() == "'/'".to_string() {
                    op_vec.push(Operator::Divide);
                } else {
                    panic!("Unknown Primitive!");
                }
                // name expression
                expr_name = graph.clone()[root_node].properties["name"].to_string();

                // now for the more complicated part, getting the arguments
                for node1 in graph.neighbors_directed(node, Outgoing) {
                    if graph.clone()[node1].properties["name"].to_string() == "'USub'".to_string() {
                        // this means there is a '-' outside one of the arguments and we need to traverse
                        // a node deeper to get the argument
                        for node2 in graph.neighbors_directed(node1, Outgoing) {
                            if graph.clone()[node2].labels == ["Opi"] {
                                let arg_string = format!(
                                    "-{}",
                                    graph.clone()[node2].properties["name"].to_string()
                                );
                                args_vec.push(Expr::Atom(Atom::Identifier(arg_string)));
                            } else {
                                panic!("Unsupported edge case where 'USub' preceeds something besides an 'Opi'!");
                            }
                        }
                    } else if graph.clone()[node1].labels == ["Primitive"] {
                        // this is the case where there are more operators and will likely require a recursive call.
                        let expr1 = tree_2_expr(graph.clone(), node1).unwrap();
                        args_vec.push(expr1);
                    } else if graph.clone()[node1].labels == ["Opi"] {
                        // nice and straight to an argument
                        args_vec.push(Expr::Atom(Atom::Identifier(
                            graph.clone()[node1].properties["name"].to_string(),
                        )));
                    } else {
                        panic!("Encoutered node that is not an 'Opi' or 'Primitive'!");
                    }
                }
            }
        }
    } else {
        if graph.clone()[root_node].labels == ["Primitive"]
            && !graph.clone()[root_node].properties["name"].to_string() == "'unpack'".to_string()
        {
            // make an operator type based on operation
            if graph.clone()[root_node].properties["name"].to_string() == "'+'".to_string() {
                op_vec.push(Operator::Add);
            } else if graph.clone()[root_node].properties["name"].to_string() == "'-'".to_string() {
                op_vec.push(Operator::Subtract);
            } else if graph.clone()[root_node].properties["name"].to_string() == "'*'".to_string() {
                op_vec.push(Operator::Multiply);
            } else if graph.clone()[root_node].properties["name"].to_string() == "'/'".to_string() {
                op_vec.push(Operator::Divide);
            } else {
                panic!("Unknown Primitive!");
            }
            // name expression
            expr_name = graph.clone()[root_node].properties["name"].to_string();

            // now for the more complicated part, getting the arguments
            for node1 in graph.neighbors_directed(root_node, Outgoing) {
                if graph.clone()[node1].properties["name"].to_string() == "'USub'".to_string() {
                    // this means there is a '-' outside one of the arguments and we need to traverse
                    // a node deeper to get the argument
                    for node2 in graph.neighbors_directed(node1, Outgoing) {
                        if graph.clone()[node2].labels == ["Opi"] {
                            let arg_string =
                                format!("-{}", graph.clone()[node2].properties["name"].to_string());
                            args_vec.push(Expr::Atom(Atom::Identifier(arg_string)));
                        } else {
                            panic!("Unsupported edge case where 'USub' preceeds something besides an 'Opi'!");
                        }
                    }
                } else if graph.clone()[node1].labels == ["Primitive"] {
                    // this is the case where there are more operators and will likely require a recursive call.
                } else if graph.clone()[node1].labels == ["Opi"] {
                    // nice and straight to an argument
                    args_vec.push(Expr::Atom(Atom::Identifier(
                        graph.clone()[node1].properties["name"].to_string(),
                    )));
                } else {
                    panic!("Encoutered node that is not an 'Opi' or 'Primitive'!");
                }
            }
        }
    }

    // now to construct the Expr
    let mut temp_expr = Expr::Expression {
        op: op_vec,
        args: args_vec,
        name: expr_name,
    };

    return Ok(temp_expr);
}

// this currently only works for un-named nodes that are not chained or have multiple incoming/outgoing edges
fn trim_un_named(
    mut graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>,
) -> petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> {
    // first create a cloned version of the graph we can modify while iterating over it.
    let mut bypass_graph = graph.clone();

    // iterate over the graph and add a new edge to bypass the un-named nodes
    for node_index in graph.clone().node_indices() {
        if graph.clone()[node_index].properties["name"].to_string() == "'un-named'".to_string() {
            let mut bypass = Vec::<NodeIndex>::new();
            for node1 in graph.neighbors_directed(node_index, Incoming) {
                bypass.push(node1);
            }
            for node2 in graph.neighbors_directed(node_index, Outgoing) {
                bypass.push(node2);
            }
            if bypass.len() == 2 {
                bypass_graph.add_edge(
                    bypass[0].clone(),
                    bypass[1].clone(),
                    graph
                        .edge_weight(graph.find_edge(node_index, bypass[1]).unwrap())
                        .unwrap()
                        .clone(),
                );
            }
        }
    }

    // now we perform a filter_map to remove the un-named nodes and only the bypass edge will remain to connect the nodes
    // we also remove the unpack node if it is present here as well
    let trimmed_graph = bypass_graph.filter_map(
        |node_index, _edge_index| {
            if !(bypass_graph.clone()[node_index].properties["name"].to_string()
                == "'un-named'".to_string()
                || bypass_graph.clone()[node_index].properties["name"].to_string()
                    == "'unpack'".to_string())
            {
                Some(graph[node_index].clone())
            } else {
                None
            }
        },
        |_node_index, edge_index| Some(edge_index.clone()),
    );

    return trimmed_graph;
}

fn subgraph_wiring(
    module_id: i64,
) -> Result<petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship>, MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some("localhost".to_string()),
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
        nodes.push(n1.clone());
    }

    // add the edges to the petgraph
    for edge in edge_list {
        let mut src = Vec::<NodeIndex>::new();
        let mut tgt = Vec::<NodeIndex>::new();
        for node_idx in nodes.clone() {
            if graph[node_idx].id == edge.start_id {
                src.push(node_idx.clone());
            }
            if graph[node_idx].id == edge.end_id {
                tgt.push(node_idx.clone());
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0].clone(), tgt[0].clone(), edge);
        }
    }
    return Ok(graph);
}

fn subgraph2petgraph(
    module_id: i64,
) -> petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> {
    let (x, y) = get_subgraph(module_id).unwrap();

    // Create a petgraph graph
    let mut graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> = Graph::new();

    // Add nodes to the petgraph graph and collect their indexes
    let mut nodes = Vec::<NodeIndex>::new();
    for node in x {
        let n1 = graph.add_node(node);
        nodes.push(n1.clone());
    }

    // add the edges to the petgraph
    for edge in y {
        let mut src = Vec::<NodeIndex>::new();
        let mut tgt = Vec::<NodeIndex>::new();
        for node_idx in nodes.clone() {
            if graph[node_idx].id == edge.start_id {
                src.push(node_idx.clone());
            }
            if graph[node_idx].id == edge.end_id {
                tgt.push(node_idx.clone());
            }
        }
        if (src.len() + tgt.len()) == 2 {
            graph.add_edge(src[0].clone(), tgt[0].clone(), edge);
        }
    }
    return graph;
}

pub fn get_subgraph(module_id: i64) -> Result<(Vec<Node>, Vec<Relationship>), MgError> {
    // construct the query that will delete the module with a given unique identifier

    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some("localhost".to_string()),
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

    return Ok((node_list, edge_list));
}
