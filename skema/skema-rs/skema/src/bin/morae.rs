use petgraph::dot::{Config, Dot};
use petgraph::matrix_graph::IndexType;
use petgraph::prelude::*;
use petgraph::visit::Dfs;
use petgraph::*;
use rsmgclient::{ConnectParams, Connection, MgError, Node, Relationship, Value};

fn main() {
    let module_id = 460;
    let graph = subgraph2petgraph(module_id);

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

    // debugging outputs
    /*for node_idx in expressions_wiring[1].clone().node_indices() {
        if expressions_wiring[1].clone()[node_idx].properties["name"].to_string()
            == "'un-named'".to_string()
        {
            println!("{:?}", node_idx);
        }
    }*/
    for i in 0..trimmed_expressions_wiring.len() {
        println!("{:?}", graph[expression_nodes[i]].id);
        println!(
            "Nodes in wiring subgraph: {}",
            trimmed_expressions_wiring[i].node_count()
        );
        println!(
            "Edges in wiring subgraph: {}",
            trimmed_expressions_wiring[i].edge_count()
        );
    }
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
    let trimmed_graph = bypass_graph.filter_map(
        |node_index, edge_index| {
            if !(bypass_graph.clone()[node_index].properties["name"].to_string()
                == "'un-named'".to_string())
            {
                Some(graph[node_index].clone())
            } else {
                None
            }
        },
        |node_index, edge_index| Some(edge_index.clone()),
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
