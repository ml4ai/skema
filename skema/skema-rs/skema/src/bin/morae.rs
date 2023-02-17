use petgraph::prelude::*;
use petgraph::*;
use rsmgclient::{ConnectParams, Connection, MgError, Node, Relationship, Value};
use skema::{
    database::{execute_query, parse_gromet_queries},
    Gromet, ModuleCollection,
};

fn main() {
    let (x, y) = test().unwrap();
    println!("nodes: {:?}, edges: {:?}", x.len(), y.len());
    //println!("{:?}", x[0].clone());

    // Create a petgraph graph
    let mut graph: petgraph::Graph<rsmgclient::Node, rsmgclient::Relationship> = Graph::new();

    // Add nodes to the petgraph graph and collect their indexes
    let mut nodes = Vec::<NodeIndex>::new();
    for node in x {
        // println!("{:?}", node.id.clone());
        let n1 = graph.add_node(node);
        nodes.push(n1.clone());
    }

    //println!("{:?}", graph[nodes[2]].id);

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

    // Use petgraph functions to analyze and visualize the graph
    println!("Number of nodes: {}", graph.node_count());
    println!("Number of edges: {}", graph.edge_count());
}

pub fn test() -> Result<(Vec<Node>, Vec<Relationship>), MgError> {
    // construct the query that will delete the module with a given unique identifier

    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some("localhost".to_string()),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // create query for nodes
    let query1 = format!(
        "MATCH p = (n)-[r*]->(m) WHERE id(n) = 0
    WITH reduce(output = [], n IN nodes(p) | output + n ) AS nodes1
    UNWIND nodes1 AS nodes2
    WITH DISTINCT nodes2
    return collect(nodes2)"
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
        "MATCH p = (n)-[r*]->(m) WHERE id(n) = 0
        WITH reduce(output = [], n IN relationships(p) | output + n ) AS edges1
        UNWIND edges1 AS edges2
        WITH DISTINCT edges2
        return collect(edges2)"
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

    //println!("{:?}", node_list.clone().len());
    //println!("{:?}", edge_list.clone().len());

    return Ok((node_list, edge_list));
}
