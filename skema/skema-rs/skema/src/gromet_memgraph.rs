use rsmgclient::{ConnectParams, Connection, MgError, Value};

use crate::FunctionType;
use crate::Gromet;
use crate::{Attribute, FunctionNet, GrometBox};
use std::process::Termination;

pub enum NodeType {
    Function,
    Predicate,
    Primitive,
    Module,
    Expression,
    Literal,
    Opo,
    Opi,
    Metadata,
}

pub enum EdgeType {
    Metadata,
    Contains,
    Of,
    Wire,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub n_type: String,
    pub value: Option<String>,
    pub name: Option<String>,
    pub node_id: String,
    pub out_idx: Option<Vec<u32>>, // opo or pof index, directly matching the wire src/tgt notation
    pub in_indx: Option<Vec<u32>>, // opi or pif index
    pub contents: u32, // This indexes which index this node has inside the attribute list,
    pub nbox: u8,
    // will be used for wiring between attribute level boxes, namely wff at the module level
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub src: String,
    pub tgt: String,
    pub e_type: String,
    pub prop: Option<u32>, // option because of opo's and opi's
}

pub fn execute_query(query: &str) -> Result<(), MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some(String::from("localhost")),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // Create simple graph.
    connection.execute_without_results(query)?;

    // Fetch the graph.
    /*let columns = connection.execute("MATCH (n)-[r]->(m) RETURN n, r, m;", None)?;
    println!("Columns: {}", columns.join(", "));
    for record in connection.fetchall()? {
        for value in record.values {
            match value {
                Value::Node(node) => print!("{}", node),
                Value::Relationship(edge) => print!("-{}-", edge),
                value => print!("{}", value),
            }
        }
        println!();
    }*/
    connection.commit()?;

    Ok(())
}

fn create_metadata_node(gromet: &Gromet) -> Vec<String> {
    let mut queries: Vec<String> = vec![];

    let mut query = String::from("CREATE");
    let collection_str = format!("{:?}", gromet.metadata_collection.as_ref().unwrap());
    let prop = format!(" (meta:Metadata {{collection: {:?}}})", collection_str);
    query.push_str(&prop);
    queries.push(query);
    return queries;
}

fn create_module(gromet: &Gromet) -> Vec<String> {
    let mut queries: Vec<String> = vec![];

    let create = String::from("CREATE");

    let node_label = String::from("mod:Module");

    let schema = format!("schema:{:?}", gromet.schema);
    let schema_version = format!("schema_version:{:?}", gromet.schema_version);
    let filename = format!("filename:{:?}", gromet.name);
    let name = format!("name:{:?}", gromet.r#fn.b[0].name.as_ref().unwrap());

    let node_query = format!(
        "{} ({} {{{},{},{},{}}})",
        create, node_label, schema, schema_version, filename, name
    );
    queries.push(node_query);

    let metadata_con = format!("{} (mod)-[mod1:Metadata]->(meta)", create);
    queries.push(metadata_con);

    let metadata_con_prop = format!(
        "set mod1.index={:?}",
        gromet.r#fn.b[0].metadata.as_ref().unwrap()
    );
    queries.push(metadata_con_prop);

    return queries;
}

fn create_function_net(gromet: &Gromet, mut start: u32) -> Vec<String> {
    // intialize the vectors
    let mut queries: Vec<String> = vec![];
    let mut nodes: Vec<Node> = vec![];
    let mut edges: Vec<Edge> = vec![];

    // iterate through module level sub-boxes and construct nodes
    // need to completely move down each branch
    // fully internally wire each branch
    let mut bf_counter: u8 = 1;
    for boxf in gromet.r#fn.bf.as_ref().unwrap().iter() {
        // construct the sub module level boxes along with their metadata and connection to module
        match boxf.function_type {
            FunctionType::Literal => {
                // first we find the pof value for the literal box
                let mut pof: Vec<u32> = vec![];
                if !gromet.r#fn.pof.clone().is_none() {
                    let mut po_idx: u32 = 1;
                    for port in gromet.r#fn.pof.clone().unwrap().iter() {
                        if port.r#box == bf_counter {
                            pof.push(po_idx);
                        } else {
                        }
                        po_idx += 1;
                    }
                } else {
                }
                let n1 = Node {
                    n_type: String::from("Literal"),
                    value: Some(format!("{:?}", boxf.value.clone().as_ref().unwrap())),
                    name: Some(format!("Literal")),
                    node_id: format!("n{}", start),
                    out_idx: Some(pof),
                    in_indx: None,
                    contents: 0,
                    nbox: bf_counter,
                };
                let e1 = Edge {
                    src: String::from("mod"),
                    tgt: format!("n{}", start),
                    e_type: String::from("Contains"),
                    prop: boxf.contents,
                };
                if !boxf.metadata.as_ref().is_none() {
                    let e2 = Edge {
                        src: format!("n{}", start),
                        tgt: String::from("meta"),
                        e_type: String::from("Metadata"),
                        prop: boxf.metadata,
                    };
                    edges.push(e2);
                } else {
                }
                nodes.push(n1.clone());
                edges.push(e1);
            }
            FunctionType::Predicate => {
                let n1 = Node {
                    n_type: String::from("Predicate"),
                    value: None,
                    name: Some(format!("Predicate{}", start)),
                    node_id: format!("n{}", start),
                    out_idx: None,
                    in_indx: None,
                    contents: boxf.contents.unwrap(),
                    nbox: bf_counter,
                };
                let e1 = Edge {
                    src: String::from("mod"),
                    tgt: format!("n{}", start),
                    e_type: String::from("Contains"),
                    prop: boxf.contents,
                };
                if !boxf.metadata.as_ref().is_none() {
                    let e2 = Edge {
                        src: format!("n{}", start),
                        tgt: String::from("meta"),
                        e_type: String::from("Metadata"),
                        prop: boxf.metadata,
                    };
                    edges.push(e2);
                } else {
                }
                nodes.push(n1.clone());
                edges.push(e1);

                // now travel to contents index of the attribute list (note it is 1 index,
                // so contents=1 => attribute[0])
                // create nodes and edges for this entry, include opo's and opi's
                start += 1;
                let idx = boxf.contents.unwrap() - 1;

                let eboxf = gromet.attributes[idx as usize].clone();
                // construct opo nodes, if not none
                if !eboxf.value.opo.clone().is_none() {
                    // grab name which is one level up and based on indexing
                    let mut opo_name = "un-named";
                    for port in gromet.r#fn.pof.as_ref().unwrap().iter() {
                        if port.r#box == bf_counter {
                            if !port.name.is_none() {
                                opo_name = port.name.as_ref().unwrap();
                            }
                        }
                    }
                    let mut oport: u32 = 0;
                    for _op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
                        let n2 = Node {
                            n_type: String::from("Opo"),
                            value: None,
                            name: Some(String::from(opo_name)),
                            node_id: format!("n{}", start),
                            out_idx: Some([oport + 1].to_vec()),
                            in_indx: None,
                            contents: idx + 1,
                            nbox: bf_counter,
                        };
                        nodes.push(n2.clone());
                        // construct edge: expression -> Opo
                        let e3 = Edge {
                            src: n1.node_id.clone(),
                            tgt: n2.node_id.clone(),
                            e_type: String::from("Port_Of"),
                            prop: None,
                        };
                        edges.push(e3);
                        // construct any metadata edges
                        if !eboxf.value.opo.clone().as_ref().unwrap()[oport as usize]
                            .metadata
                            .clone()
                            .as_ref()
                            .is_none()
                        {
                            let e4 = Edge {
                                src: n2.node_id.clone(),
                                tgt: String::from("meta"),
                                e_type: String::from("Metadata"),
                                prop: eboxf.value.opo.clone().unwrap()[oport as usize]
                                    .metadata
                                    .clone(),
                            };
                            edges.push(e4);
                        } else {
                        }
                        start += 1;
                        oport += 1;
                    }
                }
                // construct opi nodes, in not none
                if !eboxf.value.opi.clone().is_none() {
                    // grab name which is NOT one level up as in opo and based on indexing
                    let mut opi_name = "un-named";
                    for port in eboxf.value.opi.as_ref().unwrap().iter() {
                        if port.r#box == bf_counter {
                            if !port.name.is_none() {
                                opi_name = port.name.as_ref().unwrap();
                            }
                        }
                    }
                    let mut iport: u32 = 0;
                    for _op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
                        let n2 = Node {
                            n_type: String::from("Opi"),
                            value: None,
                            name: Some(String::from(opi_name)), // I think this naming will get messed up if there are multiple ports...
                            node_id: format!("n{}", start),
                            out_idx: None,
                            in_indx: Some([iport + 1].to_vec()),
                            contents: idx + 1,
                            nbox: bf_counter,
                        };
                        nodes.push(n2.clone());
                        // construct edge: expression -> Opo
                        let e3 = Edge {
                            src: n2.node_id.clone(),
                            tgt: n1.node_id.clone(),
                            e_type: String::from("Port_Of"),
                            prop: None,
                        };
                        // construct metadata edge
                        if !eboxf.value.opi.clone().as_ref().unwrap()[iport as usize]
                            .metadata
                            .as_ref()
                            .is_none()
                        {
                            let e4 = Edge {
                                src: n2.node_id,
                                tgt: String::from("meta"),
                                e_type: String::from("Metadata"),
                                prop: eboxf.value.opi.clone().unwrap()[iport as usize].metadata,
                            };
                            edges.push(e4);
                        }
                        edges.push(e3);
                        start += 1;
                        iport += 1;
                    }
                }
                // now to construct the nodes inside the expression, Literal and Primitives
                let mut box_counter: u8 = 1;
                for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
                    match sboxf.function_type {
                        FunctionType::Literal => {
                            (nodes, edges) = create_att_literal(
                                eboxf.clone(),
                                sboxf.clone(),
                                nodes.clone(),
                                edges.clone(),
                                n1.clone(),
                                idx.clone(),
                                box_counter.clone(),
                                bf_counter.clone(),
                                start.clone(),
                            );
                        }
                        FunctionType::Primitive => {
                            (nodes, edges) = create_att_primitive(
                                eboxf.clone(),
                                sboxf.clone(),
                                nodes.clone(),
                                edges.clone(),
                                n1.clone(),
                                idx.clone(),
                                box_counter.clone(),
                                bf_counter.clone(),
                                start.clone(),
                            );
                        }
                        _ => {}
                    }
                    box_counter += 1;
                    start += 1;
                }
                // Now we perform the internal wiring of this branch
                edges = internal_wiring(
                    eboxf.clone(),
                    nodes.clone(),
                    edges,
                    idx.clone(),
                    bf_counter.clone(),
                );
            }
            FunctionType::Expression => {
                let n1 = Node {
                    n_type: String::from("Expression"),
                    value: None,
                    name: Some(format!("Expression{}", start)),
                    node_id: format!("n{}", start),
                    out_idx: None,
                    in_indx: None,
                    contents: boxf.contents.unwrap(),
                    nbox: bf_counter,
                };
                let e1 = Edge {
                    src: String::from("mod"),
                    tgt: format!("n{}", start),
                    e_type: String::from("Contains"),
                    prop: boxf.contents,
                };
                if !boxf.metadata.as_ref().is_none() {
                    let e2 = Edge {
                        src: format!("n{}", start),
                        tgt: String::from("meta"),
                        e_type: String::from("Metadata"),
                        prop: boxf.metadata,
                    };
                    edges.push(e2);
                }
                nodes.push(n1.clone());
                edges.push(e1);

                // now travel to contents index of the attribute list (note it is 1 index,
                // so contents=1 => attribute[0])
                // create nodes and edges for this entry, include opo's and opi's
                start += 1;
                let idx = boxf.contents.unwrap() - 1;

                let eboxf = gromet.attributes[idx as usize].clone();
                // construct opo nodes, if not none
                if !eboxf.value.opo.clone().is_none() {
                    // grab name which is one level up and based on indexing
                    let mut opo_name = "un-named";
                    for port in gromet.r#fn.pof.as_ref().unwrap().iter() {
                        if port.r#box == bf_counter {
                            if !port.name.is_none() {
                                opo_name = port.name.as_ref().unwrap();
                            }
                        }
                    }
                    let mut oport: u32 = 0;
                    for _op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
                        let n2 = Node {
                            n_type: String::from("Opo"),
                            value: None,
                            name: Some(String::from(opo_name)),
                            node_id: format!("n{}", start),
                            out_idx: Some([oport + 1].to_vec()),
                            in_indx: None,
                            contents: idx + 1,
                            nbox: bf_counter,
                        };
                        nodes.push(n2.clone());
                        // construct edge: expression -> Opo
                        let e3 = Edge {
                            src: n1.node_id.clone(),
                            tgt: n2.node_id.clone(),
                            e_type: String::from("Port_Of"),
                            prop: None,
                        };
                        edges.push(e3);
                        // construct any metadata edges
                        if !eboxf.value.opo.clone().as_ref().unwrap()[oport as usize]
                            .metadata
                            .clone()
                            .as_ref()
                            .is_none()
                        {
                            let e4 = Edge {
                                src: n2.node_id.clone(),
                                tgt: String::from("meta"),
                                e_type: String::from("Metadata"),
                                prop: eboxf.value.opo.clone().unwrap()[oport as usize]
                                    .metadata
                                    .clone(),
                            };
                            edges.push(e4);
                        } else {
                        }
                        start += 1;
                        oport += 1;
                    }
                } else {
                }
                // construct opi nodes, in not none
                if !eboxf.value.opi.clone().is_none() {
                    // grab name which is NOT one level up as in opo and based on indexing
                    let mut opi_name = "un-named";
                    for port in eboxf.value.opi.as_ref().unwrap().iter() {
                        if port.r#box == bf_counter {
                            if !port.name.is_none() {
                                opi_name = port.name.as_ref().unwrap();
                            }
                        }
                    }
                    let mut iport: u32 = 0;
                    for _op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
                        let n2 = Node {
                            n_type: String::from("Opi"),
                            value: None,
                            name: Some(String::from(opi_name)), // I think this naming will get messed up if there are multiple ports...
                            node_id: format!("n{}", start),
                            out_idx: None,
                            in_indx: Some([iport + 1].to_vec()),
                            contents: idx + 1,
                            nbox: bf_counter,
                        };
                        nodes.push(n2.clone());
                        // construct edge: expression -> Opo
                        let e3 = Edge {
                            src: n2.node_id.clone(),
                            tgt: n1.node_id.clone(),
                            e_type: String::from("Port_Of"),
                            prop: None,
                        };
                        // construct metadata edge
                        if !eboxf.value.opi.clone().as_ref().unwrap()[iport as usize]
                            .metadata
                            .as_ref()
                            .is_none()
                        {
                            let e4 = Edge {
                                src: n2.node_id,
                                tgt: String::from("meta"),
                                e_type: String::from("Metadata"),
                                prop: eboxf.value.opi.clone().unwrap()[iport as usize].metadata,
                            };
                            edges.push(e4);
                        }
                        edges.push(e3);
                        start += 1;
                        iport += 1;
                    }
                }
                // now to construct the nodes inside the expression, Literal and Primitives
                let mut box_counter: u8 = 1;
                for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
                    match sboxf.function_type {
                        FunctionType::Literal => {
                            (nodes, edges) = create_att_literal(
                                eboxf.clone(),
                                sboxf.clone(),
                                nodes.clone(),
                                edges.clone(),
                                n1.clone(),
                                idx.clone(),
                                box_counter.clone(),
                                bf_counter.clone(),
                                start.clone(),
                            );
                        }
                        FunctionType::Primitive => {
                            (nodes, edges) = create_att_primitive(
                                eboxf.clone(),
                                sboxf.clone(),
                                nodes.clone(),
                                edges.clone(),
                                n1.clone(),
                                idx.clone(),
                                box_counter.clone(),
                                bf_counter.clone(),
                                start.clone(),
                            );
                        }
                        _ => {}
                    }
                    box_counter += 1;
                    start += 1;
                }
                // Now we perform the internal wiring of this branch
                edges = internal_wiring(
                    eboxf.clone(),
                    nodes.clone(),
                    edges,
                    idx.clone(),
                    bf_counter.clone(),
                );
            }
            FunctionType::Function => {
                let n1 = Node {
                    n_type: String::from("Function"),
                    value: None,
                    name: Some(format!("Function{}", start)),
                    node_id: format!("n{}", start),
                    out_idx: None,
                    in_indx: None,
                    contents: boxf.contents.unwrap(),
                    nbox: bf_counter,
                };
                let e1 = Edge {
                    src: String::from("mod"),
                    tgt: format!("n{}", start),
                    e_type: String::from("Contains"),
                    prop: boxf.contents,
                };
                if !boxf.metadata.as_ref().is_none() {
                    let e2 = Edge {
                        src: format!("n{}", start),
                        tgt: String::from("meta"),
                        e_type: String::from("Metadata"),
                        prop: boxf.metadata,
                    };
                    edges.push(e2);
                } else {
                }
                nodes.push(n1.clone());
                edges.push(e1);

                // now travel to contents index of the attribute list (note it is 1 index,
                // so contents=1 => attribute[0])
                // create nodes and edges for this entry, include opo's and opi's
                start += 1;
                let idx = boxf.contents.unwrap() - 1;
                // opi not wiring to internal literal
                let eboxf = gromet.attributes[idx as usize].clone();
                // construct opo nodes, if not none
                if !eboxf.value.opo.clone().is_none() {
                    // grab name which is one level up and based on indexing
                    let mut opo_name = "un-named";
                    for port in gromet.r#fn.pof.as_ref().unwrap().iter() {
                        if port.r#box == bf_counter {
                            if !port.name.is_none() {
                                opo_name = port.name.as_ref().unwrap();
                            }
                        }
                    }
                    let mut oport: u32 = 0;
                    for _op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
                        let n2 = Node {
                            n_type: String::from("Opo"),
                            value: None,
                            name: Some(String::from(opo_name)),
                            node_id: format!("n{}", start),
                            out_idx: Some([oport + 1].to_vec()),
                            in_indx: None,
                            contents: idx + 1,
                            nbox: bf_counter,
                        };
                        nodes.push(n2.clone());
                        // construct edge: expression -> Opo
                        let e3 = Edge {
                            src: n1.node_id.clone(),
                            tgt: n2.node_id.clone(),
                            e_type: String::from("Port_Of"),
                            prop: None,
                        };
                        edges.push(e3);
                        // construct any metadata edges
                        if !eboxf.value.opo.clone().as_ref().unwrap()[oport as usize]
                            .metadata
                            .clone()
                            .as_ref()
                            .is_none()
                        {
                            let e4 = Edge {
                                src: n2.node_id.clone(),
                                tgt: String::from("meta"),
                                e_type: String::from("Metadata"),
                                prop: eboxf.value.opo.clone().unwrap()[oport as usize]
                                    .metadata
                                    .clone(),
                            };
                            edges.push(e4);
                        } else {
                        }
                        start += 1;
                        oport += 1;
                    }
                } else {
                }
                // construct opi nodes, in not none
                if !eboxf.value.opi.clone().is_none() {
                    // grab name which is NOT one level up as in opo and based on indexing
                    let mut opi_name = "un-named";
                    let mut port_count: usize = 0;
                    for port in eboxf.value.opi.as_ref().unwrap().iter() {
                        if gromet.r#fn.pif.as_ref().unwrap()[port_count].r#box.clone() == bf_counter
                        {
                            if !port.name.is_none() {
                                opi_name = port.name.as_ref().unwrap();
                            }
                        }
                        port_count += 1;
                    }
                    let mut iport: u32 = 0;
                    for _op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
                        let n2 = Node {
                            n_type: String::from("Opi"),
                            value: None,
                            name: Some(String::from(opi_name)), // I think this naming will get messed up if there are multiple ports...
                            node_id: format!("n{}", start),
                            out_idx: None,
                            in_indx: Some([iport + 1].to_vec()),
                            contents: idx + 1,
                            nbox: bf_counter,
                        };
                        nodes.push(n2.clone());
                        // construct edge: expression -> Opo
                        let e3 = Edge {
                            src: n2.node_id.clone(),
                            tgt: n1.node_id.clone(),
                            e_type: String::from("Port_Of"),
                            prop: None,
                        };
                        // construct metadata edge
                        if !eboxf.value.opi.clone().as_ref().unwrap()[iport as usize]
                            .metadata
                            .as_ref()
                            .is_none()
                        {
                            let e4 = Edge {
                                src: n2.node_id,
                                tgt: String::from("meta"),
                                e_type: String::from("Metadata"),
                                prop: eboxf.value.opi.clone().unwrap()[iport as usize].metadata,
                            };
                            edges.push(e4);
                        } else {
                        }
                        edges.push(e3);
                        start += 1;
                        iport += 1;
                    }
                }
                // now to construct the nodes inside the function, currently supported Literals and Primitives
                // first include an Expression for increased depth
                let mut box_counter: u8 = 1;
                for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
                    match sboxf.function_type {
                        FunctionType::Predicate => {
                            (nodes, edges, start) = create_att_predicate(
                                &gromet.clone(),
                                eboxf.clone(),
                                sboxf.clone(),
                                nodes.clone(),
                                edges.clone(),
                                n1.clone(),
                                idx.clone(),
                                box_counter.clone(),
                                bf_counter.clone(),
                                start.clone(),
                            );
                        }
                        FunctionType::Expression => {
                            (nodes, edges, start) = create_att_expression(
                                &gromet.clone(),
                                eboxf.clone(),
                                sboxf.clone(),
                                nodes.clone(),
                                edges.clone(),
                                n1.clone(),
                                idx.clone(),
                                box_counter.clone(),
                                bf_counter.clone(),
                                start.clone(),
                            );
                        }
                        FunctionType::Literal => {
                            (nodes, edges) = create_att_literal(
                                eboxf.clone(),
                                sboxf.clone(),
                                nodes.clone(),
                                edges.clone(),
                                n1.clone(),
                                idx.clone(),
                                box_counter.clone(),
                                bf_counter.clone(),
                                start.clone(),
                            );
                        }
                        FunctionType::Primitive => {
                            (nodes, edges) = create_att_primitive(
                                eboxf.clone(),
                                sboxf.clone(),
                                nodes.clone(),
                                edges.clone(),
                                n1.clone(),
                                idx.clone(),
                                box_counter.clone(),
                                bf_counter.clone(),
                                start.clone(),
                            );
                        }
                        _ => {}
                    }
                    box_counter += 1;
                    start += 1;
                }

                // Now we perform the internal wiring of this branch
                edges = internal_wiring(
                    eboxf.clone(),
                    nodes.clone(),
                    edges,
                    idx.clone(),
                    bf_counter.clone(),
                );
                // perform cross attributal wiring of function
                edges = cross_att_wiring(
                    eboxf.clone(),
                    nodes.clone(),
                    edges,
                    idx.clone(),
                    bf_counter.clone(),
                );
            }
            _ => {}
        }
        start += 1;
        bf_counter += 1;
    }

    // add wires for inbetween attribute level boxes, so opo's, opi's and module level literals
    // between attributes
    // get wired through module level wff field, will require reading through node list to
    // match contents field to box field on wff entries
    edges = external_wiring(&gromet, nodes.clone(), edges);

    if !gromet.r#fn.bc.as_ref().is_none() {
        let mut cond_counter = 0;
        let temp_mod_node = Node {
            n_type: String::from("module"),
            value: None,
            name: None, // I think this naming will get messed up if there are multiple ports...
            node_id: format!("mod"),
            out_idx: None,
            in_indx: None,
            contents: 0,
            nbox: 0,
        };
        for cond in gromet.r#fn.bc.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            (nodes, edges, start) = create_conditional(
                gromet.r#fn.clone(), // This is gromet but is more generalizable based on scope
                nodes.clone(),
                edges.clone(),
                temp_mod_node.clone(),
                0,            // because top level
                cond_counter, // This indexes the conditional in the list of conditionals (bc)
                0,            // because top level
                start.clone(),
            );
            cond_counter += 1;
        }
    }
    // convert every node object into a node query
    let create = String::from("CREATE");
    for node in nodes.iter() {
        let mut name = String::from("a");
        let mut value = String::from("b");
        // println!("{:?}", node.clone());
        if node.name.is_none() {
            name = node.n_type.clone();
        } else {
            name = format!("{}", node.name.as_ref().unwrap());
        }
        if node.value.is_none() {
            value = String::from("None");
        } else {
            value = format!("{}", node.value.as_ref().unwrap());
        }
        let node_query = format!(
            "{} ({}:{} {{name:{:?},value:{:?}}})",
            create, node.node_id, node.n_type, name, value
        );
        queries.push(node_query);
    }

    // convert every edge object into an edge query
    for edge in edges.iter() {
        let edge_query = format!(
            "{} ({})-[e{}{}:{}]->({})",
            create, edge.src, edge.src, edge.tgt, edge.e_type, edge.tgt
        );
        queries.push(edge_query);

        if !edge.prop.is_none() {
            let set_query = format!("set e{}{}.index={}", edge.src, edge.tgt, edge.prop.unwrap());
            queries.push(set_query);
        }
    }
    return queries;
}
// this creates the framework for conditionals, including the conditional node, the pic and poc nodes and the cond, body_if and body_else edges
// The iterator through the conditionals will need to be outside this funtion
// currently crashing on making the conditional node
pub fn create_conditional(
    function_net: FunctionNet, // This is gromet but is more generalizable based on scope
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx_in: u32,       // This will index the attribute the conditional is in, if any
    cond_counter: u32, // This indexes the conditional in the list of conditionals (bc)
    bf_counter: u8,    // This indexes which box the conditional is under, if any
    mut start: u32,
) -> (Vec<Node>, Vec<Edge>, u32) {
    let n1 = Node {
        n_type: String::from("Conditional"),
        value: None,
        name: Some(format!("Conditional{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: idx_in + 1,
        nbox: bf_counter,
    };
    let e1 = Edge {
        src: parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(cond_counter),
    };
    if !function_net.bc.as_ref().unwrap()[cond_counter as usize]
        .metadata
        .as_ref()
        .is_none()
    {
        let e2 = Edge {
            src: format!("n{}", start),
            tgt: String::from("meta"),
            e_type: String::from("Metadata"),
            prop: function_net.bc.as_ref().unwrap()[cond_counter as usize].metadata,
        };
        edges.push(e2);
    }
    nodes.push(n1.clone());
    edges.push(e1);

    start += 1;

    // now we make the pic and poc ports and connect them to the conditional node
    if !function_net.pic.is_none() {
        let mut port_count = 1;
        for pic in function_net.pic.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut pic_name = String::from("Pic");
            if !pic.name.as_ref().is_none() {
                pic_name = pic.name.clone().unwrap();
            }
            // make the node
            // get the input ports
            let n2 = Node {
                n_type: String::from("Pic"),
                value: None,
                name: Some(pic_name),
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([port_count].to_vec()),
                contents: idx_in + 1,
                nbox: bf_counter,
            };
            let e3 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_of"),
                prop: None,
            };
            if !pic.metadata.as_ref().is_none() {
                let e4 = Edge {
                    src: format!("n{}", start),
                    tgt: String::from("meta"),
                    e_type: String::from("Metadata"),
                    prop: pic.metadata,
                };
                edges.push(e4);
            }
            nodes.push(n2.clone());
            edges.push(e3);

            port_count += 1;
            start += 1;
        }
    }
    if !function_net.poc.is_none() {
        let mut port_count = 1;
        for poc in function_net.poc.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut poc_name = String::from("Poc");
            if !poc.name.as_ref().is_none() {
                poc_name = poc.name.clone().unwrap();
            }
            // make the node
            let n3 = Node {
                n_type: String::from("Poc"),
                value: None,
                name: Some(poc_name),
                node_id: format!("n{}", start),
                out_idx: Some([port_count].to_vec()),
                in_indx: None,
                contents: idx_in + 1,
                nbox: bf_counter,
            };
            let e5 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_of"),
                prop: None,
            };
            if !poc.metadata.as_ref().is_none() {
                let e6 = Edge {
                    src: format!("n{}", start),
                    tgt: String::from("meta"),
                    e_type: String::from("Metadata"),
                    prop: poc.metadata,
                };
                edges.push(e6);
            }
            nodes.push(n3.clone());
            edges.push(e5);

            port_count += 1;
            start += 1;
        }
    }

    // Now let's create the edges for the conditional condition and it's bodies
    // find the nodes
    // the contents is the node's attribute reference, so need to pull off of box's contents value
    let mut condition_id = String::from("temp");
    let mut body_if_id = String::from("temp");
    let mut body_else_id = String::from("temp");
    let cond_box = function_net.bc.as_ref().unwrap()[cond_counter as usize]
        .condition
        .clone()
        .unwrap();
    let body_if_box = function_net.bc.as_ref().unwrap()[cond_counter as usize]
        .body_if
        .clone()
        .unwrap();
    let body_else_box = function_net.bc.as_ref().unwrap()[cond_counter as usize]
        .body_else
        .clone()
        .unwrap();
    for node in nodes.clone() {
        // make sure the node is in the same attribute as conditional
        if node.contents.clone()
            == function_net.bf.as_ref().unwrap()[(cond_box.clone() - 1) as usize]
                .contents
                .clone()
                .unwrap()
        {
            // make sure box matches correctly
            if (node.nbox.clone() as u32) == cond_box.clone() {
                // make sure we don't pick up ports by only getting the predicate
                if node.n_type == String::from("Predicate") {
                    condition_id = node.node_id.clone();
                }
            }
        }
        if node.contents.clone()
            == function_net.bf.as_ref().unwrap()[(body_if_box.clone() - 1) as usize]
                .contents
                .clone()
                .unwrap()
        {
            // make sure box matches correctly
            if (node.nbox.clone() as u32) == body_if_box.clone() {
                // make sure we don't pick up ports by only getting the predicate
                if node.n_type == String::from("Function") {
                    body_if_id = node.node_id.clone();
                }
            }
        }
        if node.contents.clone()
            == function_net.bf.as_ref().unwrap()[(body_else_box.clone() - 1) as usize]
                .contents
                .clone()
                .unwrap()
        {
            // make sure box matches correctly
            if (node.nbox.clone() as u32) == body_else_box.clone() {
                // make sure we don't pick up ports by only getting the predicate
                if node.n_type == String::from("Function") {
                    body_else_id = node.node_id.clone();
                }
            }
        }
    }

    // now we construct the edges
    let e7 = Edge {
        src: n1.node_id.clone(),
        tgt: condition_id,
        e_type: String::from("Condition"),
        prop: None,
    };
    edges.push(e7);
    let e8 = Edge {
        src: n1.node_id.clone(),
        tgt: body_if_id,
        e_type: String::from("Body_if"),
        prop: None,
    };
    edges.push(e8);
    let e9 = Edge {
        src: n1.node_id.clone(),
        tgt: body_else_id,
        e_type: String::from("Body_else"),
        prop: None,
    };
    edges.push(e9);

    // Now we start to wire these objects together there are two unique wire types and implicit wires that need to be made
    for wire in function_net.wfc.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_att = idx_in + 1; // attribute index of submodule (also opi contents value)
        let src_nbox = bf_counter; // nbox value of src opi
        let src_pic_idx = src_idx;

        let tgt_idx = wire.tgt; // port index
        let tgt_pof = function_net.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
        let tgt_opo_idx = tgt_pof.id.unwrap().clone(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
        let tgt_box = tgt_pof.r#box.clone(); // tgt sub module box number

        let tgt_att = function_net.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
            .contents
            .unwrap()
            .clone(); // attribute index of submodule (also opo contents value)
        let tgt_nbox = tgt_box; // nbox value of tgt opo
                                // collect information to identify the opo src node

        // now to construct the wire
        let mut wfc_src_tgt: Vec<String> = vec![];
        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if src_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_att == node.contents {
                    // only opo's
                    if node.n_type == "Pic" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_pic_idx as u32) == *p {
                                wfc_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        // missing the opo
        println!(
            "tgt_nbox: {}, tgt_att: {}, tgt_opo_idx: {}",
            tgt_nbox, tgt_att, tgt_opo_idx
        );
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_opo_idx as u32) == *p {
                                wfc_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        println!("{:?}", wfc_src_tgt);
        if wfc_src_tgt.len() == 2 {
            let e8 = Edge {
                src: wfc_src_tgt[0].clone(),
                tgt: wfc_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e8);
        }
    }
    // now to perform the wl_cargs wiring which is a connection from the condition's pof/opo to the pic

    return (nodes, edges, start);
}

pub fn create_att_expression(
    gromet: &Gromet,
    eeboxf: Attribute,
    ssboxf: GrometBox,
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx_in: u32,
    box_counter: u8,
    bf_counter: u8,
    mut start: u32,
) -> (Vec<Node>, Vec<Edge>, u32) {
    let n1 = Node {
        n_type: String::from("Expression"),
        value: None,
        name: Some(format!("Expression{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: idx_in + 1,
        nbox: bf_counter,
    };
    let e1 = Edge {
        src: parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(idx_in + 1),
    };
    if !ssboxf.metadata.as_ref().is_none() {
        let e2 = Edge {
            src: format!("n{}", start),
            tgt: String::from("meta"),
            e_type: String::from("Metadata"),
            prop: ssboxf.metadata,
        };
        edges.push(e2);
    }
    nodes.push(n1.clone());
    edges.push(e1);

    // now travel to contents index of the attribute list (note it is 1 index,
    // so contents=1 => attribute[0])
    // create nodes and edges for this entry, include opo's and opi's
    start += 1;
    let idx = ssboxf.contents.unwrap() - 1;

    let eboxf = gromet.attributes[idx as usize].clone();
    // construct opo nodes, if not none
    if !eboxf.value.opo.clone().is_none() {
        // grab name which is one level up and based on indexing /* not supported yet */
        let mut opo_name = "un-named";
        /*for port in gromet.r#fn.pof.as_ref().unwrap().iter() {
            if port.r#box == bf_counter {
                if !port.name.is_none() {
                    opo_name = port.name.as_ref().unwrap();
                }
            }
        }*/
        let mut oport: u32 = 0;
        for _op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opo"),
                value: None,
                name: Some(String::from(opo_name)),
                node_id: format!("n{}", start),
                out_idx: Some([oport + 1].to_vec()),
                in_indx: None,
                contents: idx + 1,
                nbox: bf_counter,
            };
            nodes.push(n2.clone());
            // construct edge: expression -> Opo
            let e3 = Edge {
                src: n1.node_id.clone(),
                tgt: n2.node_id.clone(),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            edges.push(e3);
            // construct any metadata edges
            if !eboxf.value.opo.clone().as_ref().unwrap()[oport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_none()
            {
                let e4 = Edge {
                    src: n2.node_id.clone(),
                    tgt: String::from("meta"),
                    e_type: String::from("Metadata"),
                    prop: eboxf.value.opo.clone().unwrap()[oport as usize]
                        .metadata
                        .clone(),
                };
                edges.push(e4);
            } else {
            }
            start += 1;
            oport += 1;
        }
    }
    // construct opi nodes, in not none
    if !eboxf.value.opi.clone().is_none() {
        // grab name which is NOT one level up as in opo and based on indexing
        let mut opi_name = "un-named";
        for port in eboxf.value.opi.as_ref().unwrap().iter() {
            if port.r#box == bf_counter {
                if !port.name.is_none() {
                    opi_name = port.name.as_ref().unwrap();
                }
            }
        }
        let mut iport: u32 = 0;
        for _op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opi"),
                value: None,
                name: Some(String::from(opi_name)), // I think this naming will get messed up if there are multiple ports...
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([iport + 1].to_vec()),
                contents: idx + 1,
                nbox: bf_counter,
            };
            nodes.push(n2.clone());
            // construct edge: expression -> Opo
            let e3 = Edge {
                src: n2.node_id.clone(),
                tgt: n1.node_id.clone(),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            // construct metadata edge
            if !eboxf.value.opi.clone().as_ref().unwrap()[iport as usize]
                .metadata
                .as_ref()
                .is_none()
            {
                let e4 = Edge {
                    src: n2.node_id,
                    tgt: String::from("meta"),
                    e_type: String::from("Metadata"),
                    prop: eboxf.value.opi.clone().unwrap()[iport as usize].metadata,
                };
                edges.push(e4);
            }
            edges.push(e3);
            start += 1;
            iport += 1;
        }
    }
    // now to construct the nodes inside the expression, Literal and Primitives
    let mut box_counter: u8 = 1;
    for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
        match sboxf.function_type {
            FunctionType::Literal => {
                (nodes, edges) = create_att_literal(
                    eboxf.clone(),
                    sboxf.clone(),
                    nodes.clone(),
                    edges.clone(),
                    n1.clone(),
                    idx.clone(),
                    box_counter.clone(),
                    bf_counter.clone(),
                    start.clone(),
                );
            }
            FunctionType::Primitive => {
                (nodes, edges) = create_att_primitive(
                    eboxf.clone(),
                    sboxf.clone(),
                    nodes.clone(),
                    edges.clone(),
                    n1.clone(),
                    idx.clone(),
                    box_counter.clone(),
                    bf_counter.clone(),
                    start.clone(),
                );
            }
            _ => {}
        }
        box_counter += 1;
        start += 1;
    }
    // Now we perform the internal wiring of this branch
    edges = internal_wiring(
        eboxf.clone(),
        nodes.clone(),
        edges,
        idx.clone(),
        bf_counter.clone(),
    );
    return (nodes, edges, start);
}

pub fn create_att_predicate(
    gromet: &Gromet,
    eeboxf: Attribute,
    ssboxf: GrometBox,
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx_in: u32,
    box_counter: u8,
    bf_counter: u8,
    mut start: u32,
) -> (Vec<Node>, Vec<Edge>, u32) {
    let n1 = Node {
        n_type: String::from("Predicate"),
        value: None,
        name: Some(format!("Predicate{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: idx_in + 1,
        nbox: bf_counter,
    };
    let e1 = Edge {
        src: parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(idx_in + 1),
    };
    if !ssboxf.metadata.as_ref().is_none() {
        let e2 = Edge {
            src: format!("n{}", start),
            tgt: String::from("meta"),
            e_type: String::from("Metadata"),
            prop: ssboxf.metadata,
        };
        edges.push(e2);
    } else {
    }
    nodes.push(n1.clone());
    edges.push(e1);

    // now travel to contents index of the attribute list (note it is 1 index,
    // so contents=1 => attribute[0])
    // create nodes and edges for this entry, include opo's and opi's
    start += 1;
    let idx = ssboxf.contents.unwrap() - 1;

    let eboxf = gromet.attributes[idx as usize].clone();
    // construct opo nodes, if not none
    if !eboxf.value.opo.clone().is_none() {
        // grab name which is one level up and based on indexing /* not supported yet */
        let mut opo_name = "un-named";
        /*for port in gromet.r#fn.pof.as_ref().unwrap().iter() {
            if port.r#box == bf_counter {
                if !port.name.is_none() {
                    opo_name = port.name.as_ref().unwrap();
                }
            }
        }*/
        let mut oport: u32 = 0;
        for _op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opo"),
                value: None,
                name: Some(String::from(opo_name)),
                node_id: format!("n{}", start),
                out_idx: Some([oport + 1].to_vec()),
                in_indx: None,
                contents: idx + 1,
                nbox: bf_counter,
            };
            nodes.push(n2.clone());
            // construct edge: expression -> Opo
            let e3 = Edge {
                src: n1.node_id.clone(),
                tgt: n2.node_id.clone(),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            edges.push(e3);
            // construct any metadata edges
            if !eboxf.value.opo.clone().as_ref().unwrap()[oport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_none()
            {
                let e4 = Edge {
                    src: n2.node_id.clone(),
                    tgt: String::from("meta"),
                    e_type: String::from("Metadata"),
                    prop: eboxf.value.opo.clone().unwrap()[oport as usize]
                        .metadata
                        .clone(),
                };
                edges.push(e4);
            } else {
            }
            start += 1;
            oport += 1;
        }
    }
    // construct opi nodes, in not none
    if !eboxf.value.opi.clone().is_none() {
        // grab name which is NOT one level up as in opo and based on indexing
        let mut opi_name = "un-named";
        for port in eboxf.value.opi.as_ref().unwrap().iter() {
            if port.r#box == bf_counter {
                if !port.name.is_none() {
                    opi_name = port.name.as_ref().unwrap();
                }
            }
        }
        let mut iport: u32 = 0;
        for _op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
            let n3 = Node {
                n_type: String::from("Opi"),
                value: None,
                name: Some(String::from(opi_name)), // I think this naming will get messed up if there are multiple ports...
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([iport + 1].to_vec()),
                contents: idx + 1,
                nbox: bf_counter,
            };
            nodes.push(n3.clone());
            // construct edge: expression -> Opo
            let e3 = Edge {
                src: n3.node_id.clone(),
                tgt: n1.node_id.clone(),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            // construct metadata edge
            if !eboxf.value.opi.clone().as_ref().unwrap()[iport as usize]
                .metadata
                .as_ref()
                .is_none()
            {
                let e4 = Edge {
                    src: n3.node_id,
                    tgt: String::from("meta"),
                    e_type: String::from("Metadata"),
                    prop: eboxf.value.opi.clone().unwrap()[iport as usize].metadata,
                };
                edges.push(e4);
            }
            edges.push(e3);
            start += 1;
            iport += 1;
        }
    }
    // now to construct the nodes inside the predicate, Literal and Primitives
    println!("{}", start);
    let mut box_counter: u8 = 1;
    for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
        match sboxf.function_type {
            FunctionType::Literal => {
                println!("Predicate Literal: {}", start);
                (nodes, edges) = create_att_literal(
                    eboxf.clone(),
                    sboxf.clone(),
                    nodes.clone(),
                    edges.clone(),
                    n1.clone(),
                    idx.clone(),
                    box_counter.clone(),
                    bf_counter.clone(),
                    start.clone(),
                );
                start += 1;
            }
            FunctionType::Primitive => {
                println!("Predicate Primitive: {}", start);
                (nodes, edges) = create_att_primitive(
                    eboxf.clone(),
                    sboxf.clone(),
                    nodes.clone(),
                    edges.clone(),
                    n1.clone(),
                    idx.clone(),
                    box_counter.clone(),
                    bf_counter.clone(),
                    start.clone(),
                );
                start += 1;
            }
            _ => {}
        }
        box_counter += 1;
        start += 1;
    }
    // Now we perform the internal wiring of this branch
    edges = internal_wiring(
        eboxf.clone(),
        nodes.clone(),
        edges,
        idx.clone(),
        bf_counter.clone(),
    );
    return (nodes, edges, start);
}

pub fn create_att_literal(
    eboxf: Attribute,
    sboxf: GrometBox,
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx: u32,
    box_counter: u8,
    bf_counter: u8,
    start: u32,
) -> (Vec<Node>, Vec<Edge>) {
    // first find the pof's for box
    let mut pof: Vec<u32> = vec![];
    if !eboxf.value.pof.clone().is_none() {
        let mut po_idx: u32 = 1;
        for port in eboxf.value.pof.clone().unwrap().iter() {
            if port.r#box == box_counter {
                pof.push(po_idx);
            } else {
            }
            po_idx += 1;
        }
    }
    // now make the node with the port information
    let n3 = Node {
        n_type: String::from("Literal"),
        value: Some(format!("{:?}", sboxf.value.clone().as_ref().unwrap())),
        name: None,
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: None, // literals should only have out ports
        contents: idx + 1,
        nbox: bf_counter,
    };
    nodes.push(n3.clone());
    // make edge connecting to expression
    let e4 = Edge {
        src: parent_node.node_id.clone(),
        tgt: n3.node_id.clone(),
        e_type: String::from("Contains"),
        prop: None,
    };
    edges.push(e4);
    // now add a metadata edge
    if !sboxf.metadata.is_none() {
        let e5 = Edge {
            src: n3.node_id,
            tgt: String::from("meta"),
            e_type: String::from("Metadata"),
            prop: sboxf.metadata.clone(),
        };
        edges.push(e5);
    }
    return (nodes, edges);
}

pub fn create_att_primitive(
    eboxf: Attribute,
    sboxf: GrometBox,
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx: u32,
    box_counter: u8,
    bf_counter: u8,
    start: u32,
) -> (Vec<Node>, Vec<Edge>) {
    // first find the pof's for box
    let mut pof: Vec<u32> = vec![];
    if !eboxf.value.pof.clone().is_none() {
        let mut po_idx: u32 = 1;
        for port in eboxf.value.pof.clone().unwrap().iter() {
            if port.r#box == box_counter {
                pof.push(po_idx);
            }
            po_idx += 1;
        }
    } else {
    }
    // then find pif's for box
    let mut pif: Vec<u32> = vec![];
    if !eboxf.value.pif.clone().is_none() {
        let mut pi_idx: u32 = 1;
        for port in eboxf.value.pif.clone().unwrap().iter() {
            if port.r#box == box_counter {
                pif.push(pi_idx);
            }
            pi_idx += 1;
        }
    } else {
    }
    // now make the node with the port information
    let n3 = Node {
        n_type: String::from("Primitive"),
        value: None,
        name: sboxf.name.clone(),
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: Some(pif),
        contents: idx + 1,
        nbox: bf_counter,
    };
    nodes.push(n3.clone());
    // make edge connecting to expression
    let e4 = Edge {
        src: parent_node.node_id.clone(),
        tgt: n3.node_id.clone(),
        e_type: String::from("Contains"),
        prop: None,
    };
    edges.push(e4);
    // now add a metadata edge
    if !sboxf.metadata.clone().is_none() {
        let e5 = Edge {
            src: n3.node_id.clone(),
            tgt: String::from("meta"),
            e_type: String::from("Metadata"),
            prop: sboxf.metadata.clone(),
        };
        edges.push(e5);
    }
    return (nodes, edges);
}

// having issues with deeply nested structure, it is breaking in the internal wiring of the function level.
pub fn wfopi_wiring(
    eboxf: Attribute,
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,
    bf_counter: u8,
) -> Vec<Edge> {
    // iterate through all wires of type
    for wire in eboxf.value.wfopi.unwrap().iter() {
        let mut wfopi_src_tgt: Vec<String> = vec![];
        // find the src node
        // ******* missing a src node *******
        // add a box check?
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx + 1) == node.contents {
                    // only include nodes with pifs
                    if !node.in_indx.is_none() {
                        // exclude opi's
                        if node.n_type != "Opi" {
                            // iterate through port to check for src
                            for p in node.in_indx.as_ref().unwrap().iter() {
                                // push the src first, being pif
                                if (wire.src as u32) == *p {
                                    wfopi_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        // find the tgt node
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx + 1) == node.contents {
                    // only opi's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (wire.tgt as u32) == *p {
                                wfopi_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if wfopi_src_tgt.len() == 2 {
            let e6 = Edge {
                src: wfopi_src_tgt[0].clone(),
                tgt: wfopi_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e6);
        }
    }
    return edges;
}

pub fn wfopo_wiring(
    eboxf: Attribute,
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,
    bf_counter: u8,
) -> Vec<Edge> {
    // iterate through all wires of type
    for wire in eboxf.value.wfopo.unwrap().iter() {
        let mut wfopo_src_tgt: Vec<String> = vec![];
        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx + 1) == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (wire.src as u32) == *p {
                                wfopo_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        // finding the tgt node
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx + 1) == node.contents {
                    // only include nodes with pofs
                    if !node.out_idx.is_none() {
                        // exclude opo's
                        if node.n_type != "Opo" {
                            // iterate through port to check for src
                            for p in node.out_idx.as_ref().unwrap().iter() {
                                // push the tgt
                                if (wire.tgt as u32) == *p {
                                    wfopo_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        if wfopo_src_tgt.len() == 2 {
            let e7 = Edge {
                src: wfopo_src_tgt[0].clone(),
                tgt: wfopo_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e7);
        }
    }
    return edges;
}
pub fn wff_wiring(
    eboxf: Attribute,
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,
    bf_counter: u8,
) -> Vec<Edge> {
    // iterate through all wires of type
    for wire in eboxf.value.wff.unwrap().iter() {
        let mut wff_src_tgt: Vec<String> = vec![];
        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx + 1) == node.contents {
                    // only include nodes with pifs
                    if !node.in_indx.is_none() {
                        // exclude opo's
                        if node.n_type != "Opi" {
                            // iterate through port to check for src
                            for p in node.in_indx.as_ref().unwrap().iter() {
                                // push the tgt
                                if (wire.src as u32) == *p {
                                    wff_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        // finding the tgt node
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx + 1) == node.contents {
                    // only include nodes with pofs
                    if !node.out_idx.is_none() {
                        // exclude opo's
                        if node.n_type != "Opo" {
                            // iterate through port to check for tgt
                            for p in node.out_idx.as_ref().unwrap().iter() {
                                // push the tgt
                                if (wire.tgt as u32) == *p {
                                    wff_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        if wff_src_tgt.len() == 2 {
            let e8 = Edge {
                src: wff_src_tgt[0].clone(),
                tgt: wff_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e8);
        }
    }
    return edges;
}
pub fn internal_wiring(
    eboxf: Attribute,
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,
    bf_counter: u8,
) -> Vec<Edge> {
    // first lets wire the wfopi, note we need to first limit ourselves
    // to only nodes in the current attribute by checking the contents field
    // and then we run find the ports that match the wire src and tgt.
    // wfopi: pif -> opi
    // wff: pif -> pof
    // wfopo: opo -> pof

    // check if wire exists, wfopi
    if !eboxf.value.wfopi.clone().is_none() {
        edges = wfopi_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone(),
            bf_counter.clone(),
        );
    }

    // check if wire exists, wfopo
    if !eboxf.value.wfopo.is_none() {
        edges = wfopo_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone(),
            bf_counter.clone(),
        );
    }

    // check if wire exists, wff
    if !eboxf.value.wff.is_none() {
        edges = wff_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone(),
            bf_counter.clone(),
        );
    }
    return edges;
}
pub fn cross_att_wiring(
    eboxf: Attribute, // This is the current attribute
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,       // this +1 is the current attribute index
    bf_counter: u8, // this is the current box
) -> Vec<Edge> {
    // wire id corresponds to subport index in list so ex: wff src.id="1" means the first opi in the list of the src.box, this is the in_idx in the opi or out_indx in opo.
    // This will have to run wfopo wfopi and wff all in order to get the cross attribual wiring that can exist in all of them, refactoring won't do much in code saving space though.
    // for cross attributal wiring they will construct the following types of wires
    // wfopi: opi -> opi
    // wff: opi -> opo
    // wfopo: opo -> opo

    // check if wire exists, wfopi
    if !eboxf.value.wfopi.clone().is_none() {
        edges = wfopi_cross_att_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone(),
            bf_counter.clone(),
        );
    }

    // check if wire exists, wfopo
    if !eboxf.value.wfopo.is_none() {
        edges = wfopo_cross_att_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone(),
            bf_counter.clone(),
        );
    }

    // check if wire exists, wff
    if !eboxf.value.wff.is_none() {
        edges = wff_cross_att_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone(),
            bf_counter.clone(),
        );
    }
    return edges;
}
// this will construct connections from the function opi's to the sub module opi's, tracing inputs through the function
// opi(sub)->opi(fun)
pub fn wfopi_cross_att_wiring(
    eboxf: Attribute, // This is the current attribute
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,       // this +1 is the current attribute index
    bf_counter: u8, // this is the current box
) -> Vec<Edge> {
    for wire in eboxf.value.wfopi.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_pif = eboxf.value.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port
        let src_opi_idx = src_pif.id.unwrap().clone(); // index of opi port in opi list in src sub module (also opi node in_indx value)
        let src_box = src_pif.r#box.clone(); // src sub module box number

        // make sure it's a cross attributal wire and not internal
        if !eboxf.value.bf.as_ref().unwrap()[(src_box - 1) as usize]
            .contents
            .clone()
            .is_none()
        {
            let src_att = eboxf.value.bf.as_ref().unwrap()[(src_box - 1) as usize]
                .contents
                .unwrap()
                .clone(); // attribute index of submodule (also opi contents value)
            let src_nbox = bf_counter; // nbox value of src opi
                                       // collect information to identify the opi target node
            let tgt_opi_idx = wire.tgt; // index of opi port in tgt function
            let tgt_att = idx + 1; // attribute index of function
            let tgt_nbox = bf_counter; // nbox value of tgt opi

            // now to construct the wire
            let mut wfopi_src_tgt: Vec<String> = vec![];
            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if src_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if src_att == node.contents {
                        // only opo's
                        if node.n_type == "Opi" {
                            // iterate through port to check for tgt
                            for p in node.in_indx.as_ref().unwrap().iter() {
                                // push the src first, being pif
                                if (src_opi_idx as u32) == *p {
                                    wfopi_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
            for node in nodes.iter() {
                // make sure in correct box
                if tgt_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att == node.contents {
                        // only opo's
                        if node.n_type == "Opi" {
                            // iterate through port to check for tgt
                            for p in node.in_indx.as_ref().unwrap().iter() {
                                // push the src first, being pif
                                if (tgt_opi_idx as u32) == *p {
                                    wfopi_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
            if wfopi_src_tgt.len() == 2 {
                let e8 = Edge {
                    src: wfopi_src_tgt[0].clone(),
                    tgt: wfopi_src_tgt[1].clone(),
                    e_type: String::from("Wire"),
                    prop: None,
                };
                edges.push(e8);
            }
        }
    }
    return edges;
}
// this will construct connections from the function opo's to the sub module opo's, tracing outputs through the function
// opo(fun)->opo(sub)
pub fn wfopo_cross_att_wiring(
    eboxf: Attribute, // This is the current attribute
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,       // this +1 is the current attribute index
    bf_counter: u8, // this is the current box
) -> Vec<Edge> {
    for wire in eboxf.value.wfopo.as_ref().unwrap().iter() {
        // collect info to identify the opo tgt node
        let tgt_idx = wire.tgt; // port index
        let tgt_pof = eboxf.value.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
        let tgt_opo_idx = tgt_pof.id.unwrap().clone(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
        let tgt_box = tgt_pof.r#box.clone(); // tgt sub module box number

        // make sure its a cross attributal wiring and not an internal one
        if !eboxf.value.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
            .contents
            .clone()
            .is_none()
        {
            let tgt_att = eboxf.value.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .contents
                .unwrap()
                .clone(); // attribute index of submodule (also opo contents value)
            let tgt_nbox = bf_counter; // nbox value of tgt opo
                                       // collect information to identify the opo src node
            let src_opo_idx = wire.src; // index of opo port in src function
            let src_att = idx + 1; // attribute index of function
            let src_nbox = bf_counter; // nbox value of tgt opo

            // now to construct the wire
            let mut wfopo_src_tgt: Vec<String> = vec![];
            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if src_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if src_att == node.contents {
                        // only opo's
                        if node.n_type == "Opo" {
                            // iterate through port to check for tgt
                            for p in node.out_idx.as_ref().unwrap().iter() {
                                // push the src first, being pif
                                if (src_opo_idx as u32) == *p {
                                    wfopo_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
            for node in nodes.iter() {
                // make sure in correct box
                if tgt_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att == node.contents {
                        // only opo's
                        if node.n_type == "Opo" {
                            // iterate through port to check for tgt
                            for p in node.out_idx.as_ref().unwrap().iter() {
                                // push the src first, being pif
                                if (tgt_opo_idx as u32) == *p {
                                    wfopo_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
            if wfopo_src_tgt.len() == 2 {
                let e8 = Edge {
                    src: wfopo_src_tgt[0].clone(),
                    tgt: wfopo_src_tgt[1].clone(),
                    e_type: String::from("Wire"),
                    prop: None,
                };
                edges.push(e8);
            }
        }
    }
    return edges;
}
// this will construct connections from the sub function modules opi's to another sub module opo's, tracing data inside the function
// opi(sub)->opo(sub)
pub fn wff_cross_att_wiring(
    eboxf: Attribute, // This is the current attribute
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,       // this +1 is the current attribute index
    bf_counter: u8, // this is the current box
) -> Vec<Edge> {
    for wire in eboxf.value.wff.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_pif = eboxf.value.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port
        let src_opi_idx = src_pif.id.unwrap().clone(); // index of opi port in opi list in src sub module (also opi node in_indx value)
        let src_box = src_pif.r#box.clone(); // src sub module box number

        // make sure its a cross attributal wiring and not an internal wire
        if !eboxf.value.bf.as_ref().unwrap()[(src_box - 1) as usize]
            .contents
            .clone()
            .is_none()
        {
            let src_att = eboxf.value.bf.as_ref().unwrap()[(src_box - 1) as usize]
                .contents
                .unwrap()
                .clone(); // attribute index of submodule (also opi contents value)
            let src_nbox = bf_counter; // nbox value of src opi
                                       // collect info to identify the opo tgt node
            let tgt_idx = wire.tgt; // port index
            let tgt_pof = eboxf.value.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
            let tgt_opo_idx = tgt_pof.id.unwrap().clone(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
            let tgt_box = tgt_pof.r#box.clone(); // tgt sub module box number

            // make sure its a cross attributal wiring and not an internal wire
            if eboxf.value.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .contents
                .clone()
                .is_none()
            {
                let tgt_att = eboxf.value.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                    .contents
                    .unwrap()
                    .clone(); // attribute index of submodule (also opo contents value)
                let tgt_nbox = bf_counter; // nbox value of tgt opo

                // now to construct the wire
                let mut wff_src_tgt: Vec<String> = vec![];
                // find the src node
                for node in nodes.iter() {
                    // make sure in correct box
                    if src_nbox == node.nbox {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if src_att == node.contents {
                            // only opo's
                            if node.n_type == "Opi" {
                                // iterate through port to check for tgt
                                for p in node.in_indx.as_ref().unwrap().iter() {
                                    // push the src first, being pif
                                    if (src_opi_idx as u32) == *p {
                                        wff_src_tgt.push(node.node_id.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                for node in nodes.iter() {
                    // make sure in correct box
                    if tgt_nbox == node.nbox {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if tgt_att == node.contents {
                            // only opo's
                            if node.n_type == "Opo" {
                                // iterate through port to check for tgt
                                for p in node.out_idx.as_ref().unwrap().iter() {
                                    // push the src first, being pif
                                    if (tgt_opo_idx as u32) == *p {
                                        wff_src_tgt.push(node.node_id.clone());
                                    }
                                }
                            }
                        }
                    }
                }
                if wff_src_tgt.len() == 2 {
                    let e8 = Edge {
                        src: wff_src_tgt[0].clone(),
                        tgt: wff_src_tgt[1].clone(),
                        e_type: String::from("Wire"),
                        prop: None,
                    };
                    edges.push(e8);
                }
            }
        }
    }
    return edges;
}
// external wiring is the wiring between boxes at the module level
pub fn external_wiring(gromet: &Gromet, nodes: Vec<Node>, mut edges: Vec<Edge>) -> Vec<Edge> {
    if !gromet.r#fn.wff.as_ref().is_none() {
        for wire in gromet.r#fn.wff.as_ref().unwrap().iter() {
            let src_idx = wire.src; // pif wire connects to
            let tgt_idx = wire.tgt; // pof wire connects to
            let src_id = gromet.r#fn.pif.as_ref().unwrap()[(src_idx - 1) as usize]
                .id
                .unwrap(); // pif id
            let src_box = gromet.r#fn.pif.as_ref().unwrap()[(src_idx - 1) as usize].r#box; // pif box
            let mut src_att = None;
            if gromet.r#fn.bf.as_ref().unwrap()[(src_box - 1) as usize]
                .function_type
                .clone()
                == FunctionType::Function
            {
                src_att = gromet.r#fn.bf.as_ref().unwrap()[(src_box - 1) as usize].contents;
            }
            let tgt_id = gromet.r#fn.pof.as_ref().unwrap()[(tgt_idx - 1) as usize]
                .id
                .unwrap(); // pof id
            let tgt_box = gromet.r#fn.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].r#box; // pof box
            let mut tgt_att = None;
            if gromet.r#fn.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .function_type
                .clone()
                == FunctionType::Function
            {
                tgt_att = gromet.r#fn.bf.as_ref().unwrap()[(tgt_box - 1) as usize].contents;
            }
            let mut wff_src_tgt: Vec<String> = vec![];
            // This is double counting since only check is name and box, check on attributes?
            // find the src
            for node in nodes.iter() {
                if node.nbox == src_box {
                    if node.n_type == "Opi" {
                        if src_att.is_none() {
                            for p in node.in_indx.as_ref().unwrap().iter() {
                                // push the src
                                if (src_id as u32) == *p {
                                    wff_src_tgt.push(node.node_id.clone());
                                }
                            }
                        } else {
                            // perform extra check to make sure not getting sub cross attributal nodes
                            if src_att.unwrap().clone() == node.contents {
                                for p in node.in_indx.as_ref().unwrap().iter() {
                                    // push the src
                                    if (src_id as u32) == *p {
                                        wff_src_tgt.push(node.node_id.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // find the tgt
            for node in nodes.iter() {
                // check this field
                if node.n_type == "Opo" {
                    if node.nbox == tgt_box {
                        if tgt_att.is_none() {
                            for p in node.out_idx.as_ref().unwrap().iter() {
                                // push the tgt
                                if (tgt_id as u32) == *p {
                                    wff_src_tgt.push(node.node_id.clone());
                                }
                            }
                        } else {
                            // perform extra check to make sure not getting sub cross attributal nodes
                            if tgt_att.unwrap().clone() == node.contents {
                                for p in node.out_idx.as_ref().unwrap().iter() {
                                    // push the tgt
                                    if (tgt_id as u32) == *p {
                                        wff_src_tgt.push(node.node_id.clone());
                                    }
                                }
                            }
                        }
                    }
                } else if node.n_type == "Literal" {
                    if node.nbox == tgt_box {
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the tgt
                            if (tgt_box as u32) == *p {
                                wff_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
            let e9 = Edge {
                src: wff_src_tgt[0].clone(),
                tgt: wff_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e9);
        }
    }
    return edges;
}

pub fn parse_gromet_queries(gromet: Gromet) -> Vec<String> {
    let mut queries: Vec<String> = vec![];

    let mut start: u32 = 0;

    queries.append(&mut create_metadata_node(&gromet));
    queries.append(&mut create_module(&gromet));
    queries.append(&mut create_function_net(&gromet, start));

    return queries;
}
