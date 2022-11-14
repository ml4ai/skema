use rsmgclient::{ConnectParams, Connection, MgError, Value};

use crate::FunctionType;
use crate::Gromet;
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
    pub out_idx: Option<Vec<u32>>, // opo or pof index, directly matching the wff, wfopo, or wfopi src/tgt notation
    pub in_indx: Option<Vec<u32>>, // opi or pif index
    pub contents: u32, // This indexes which index this node has inside the attribute list, will be used for wiring between attribute level boxes, namely wff at the module level
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
    let columns = connection.execute("MATCH (n)-[r]->(m) RETURN n, r, m;", None)?;
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
    }
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
    let mut egdes: Vec<Edge> = vec![];

    // iterate through module level sub-boxes and construct nodes
    // need to completely move down each branch
    // fully internally wire each branch
    let mut bf_counter: u8 = 1;
    for boxf in gromet.r#fn.bf.as_ref().unwrap().iter() {
        // construct first the sub module level boxes along with their metadata and connection to module
        match boxf.function_type {
            FunctionType::Expression => {
                let n1 = Node {
                    n_type: String::from("Expression"),
                    value: None,
                    name: Some(format!("Expression{}", start)),
                    node_id: format!("n{}", start),
                    out_idx: None,
                    in_indx: None,
                    contents: boxf.contents.unwrap(),
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
                    egdes.push(e2);
                } else {
                }
                nodes.push(n1.clone());
                egdes.push(e1);

                // now travel to contents index of the attribute list (note it is 1 index, so contents=1 => attribute[0])
                // create nodes and edges for this entry, include opo's and opi's
                start += 1;
                let idx = boxf.contents.unwrap() - 1;

                let eboxf = gromet.attributes[idx as usize].clone();
                // construct opo nodes, if not none
                // NOTE::: edges shouls go from opo to expression !!!!!
                if !eboxf.value.opo.clone().is_none() {
                    // grab name which is one level up and based on indexing because most frustrating that way
                    let mut opo_name = "un-named";
                    for port in gromet.r#fn.pof.as_ref().unwrap().iter() {
                        if port.r#box == bf_counter {
                            if !port.name.is_none() {
                                opo_name = port.name.as_ref().unwrap();
                            }
                        }
                    }
                    let mut oport: u32 = 0;
                    for op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
                        let n2 = Node {
                            n_type: String::from("Opo"),
                            value: None,
                            name: Some(String::from(opo_name)),
                            node_id: format!("n{}", start),
                            out_idx: Some([oport + 1].to_vec()),
                            in_indx: None,
                            contents: idx + 1,
                        };
                        nodes.push(n2.clone());
                        // construct edge: expression -> Opo
                        let e3 = Edge {
                            src: n2.node_id.clone(),
                            tgt: n1.node_id.clone(),
                            e_type: String::from("Port_Of"),
                            prop: None,
                        };
                        egdes.push(e3);
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
                            egdes.push(e4);
                        } else {
                        }
                        start += 1;
                        oport += 1;
                    }
                } else {
                }
                // construct opi nodes, in not none
                if !eboxf.value.opi.clone().is_none() {
                    let mut iport: u32 = 0;
                    for op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
                        let n2 = Node {
                            n_type: String::from("Opi"),
                            value: None,
                            name: None,
                            node_id: format!("n{}", start),
                            out_idx: None,
                            in_indx: Some([iport + 1].to_vec()),
                            contents: idx + 1,
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
                            egdes.push(e4);
                        } else {
                        }
                        egdes.push(e3);
                        start += 1;
                        iport += 1;
                    }
                } else {
                }
                // now to construct the nodes inside the expression
                let mut box_counter: u8 = 1;
                for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
                    match sboxf.function_type {
                        FunctionType::Literal => {
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
                            } else {
                            }
                            // then find pif's for box
                            let mut pif: Vec<u32> = vec![];
                            if !eboxf.value.pif.clone().is_none() {
                                let mut pi_idx: u32 = 1;
                                for port in eboxf.value.pif.clone().unwrap().iter() {
                                    if port.r#box == box_counter {
                                        pif.push(pi_idx);
                                    } else {
                                    }
                                    pi_idx += 1;
                                }
                            } else {
                            }
                            // now make the node with the port information
                            let n3 = Node {
                                n_type: String::from("Literal"),
                                value: Some(format!("{:?}", sboxf.value.clone().as_ref().unwrap())),
                                name: None,
                                node_id: format!("n{}", start),
                                out_idx: Some(pof),
                                in_indx: Some(pif),
                                contents: idx + 1,
                            };
                            nodes.push(n3.clone());
                            // make edge connecting to expression
                            let e4 = Edge {
                                src: n1.node_id.clone(),
                                tgt: n3.node_id.clone(),
                                e_type: String::from("Contains"),
                                prop: None,
                            };
                            egdes.push(e4);
                            // now add a metadata edge
                            if !sboxf.metadata.is_none() {
                                let e5 = Edge {
                                    src: n3.node_id,
                                    tgt: String::from("meta"),
                                    e_type: String::from("Metadata"),
                                    prop: sboxf.metadata.clone(),
                                };
                                egdes.push(e5);
                            } else {
                            }
                        }
                        FunctionType::Primitive => {
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
                            } else {
                            }
                            // then find pif's for box
                            let mut pif: Vec<u32> = vec![];
                            if !eboxf.value.pif.clone().is_none() {
                                let mut pi_idx: u32 = 1;
                                for port in eboxf.value.pif.clone().unwrap().iter() {
                                    if port.r#box == box_counter {
                                        pif.push(pi_idx);
                                    } else {
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
                            };
                            nodes.push(n3.clone());
                            // make edge connecting to expression
                            let e4 = Edge {
                                src: n1.node_id.clone(),
                                tgt: n3.node_id.clone(),
                                e_type: String::from("Contains"),
                                prop: None,
                            };
                            egdes.push(e4);
                            // now add a metadata edge
                            if !sboxf.metadata.clone().is_none() {
                                let e5 = Edge {
                                    src: n3.node_id.clone(),
                                    tgt: String::from("meta"),
                                    e_type: String::from("Metadata"),
                                    prop: sboxf.metadata.clone(),
                                };
                                egdes.push(e5);
                            } else {
                            }
                        }
                        _ => {}
                    }
                    box_counter += 1;
                    start += 1;
                }
                // Now we perform the internal wiring of this branch
                // first lets wire the wfopi, note we need to first limit ourselves to only nodes in the current attribute by checking the contents field and then we run find the ports that match the wire src and tgt.
                // wfopi: pif -> opi
                // wff: pif -> pof
                // wfopo: opo -> pof

                // check if wire exists, wfopi
                if !eboxf.value.wfopi.clone().is_none() {
                    // iterate through all wires of type
                    for wire in eboxf.value.wfopi.unwrap().iter() {
                        let mut wfopi_src_tgt: Vec<String> = vec![];
                        // find the src node
                        for node in nodes.iter() {
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
                        // find the tgt node
                        for node in nodes.iter() {
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
                        let e6 = Edge {
                            src: wfopi_src_tgt[0].clone(),
                            tgt: wfopi_src_tgt[1].clone(),
                            e_type: String::from("Wire"),
                            prop: None,
                        };
                        egdes.push(e6);
                    }
                }

                // check if wire exists, wfopo
                if !eboxf.value.wfopo.is_none() {
                    // iterate through all wires of type
                    for wire in eboxf.value.wfopo.unwrap().iter() {
                        let mut wfopo_src_tgt: Vec<String> = vec![];
                        // find the src node
                        for node in nodes.iter() {
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
                        // finding the tgt node
                        for node in nodes.iter() {
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
                        let e7 = Edge {
                            src: wfopo_src_tgt[0].clone(),
                            tgt: wfopo_src_tgt[1].clone(),
                            e_type: String::from("Wire"),
                            prop: None,
                        };
                        egdes.push(e7);
                    }
                }

                // check if wire exists, wff
                if !eboxf.value.wff.is_none() {
                    // iterate through all wires of type
                    for wire in eboxf.value.wff.unwrap().iter() {
                        let mut wff_src_tgt: Vec<String> = vec![];
                        // find the src node
                        for node in nodes.iter() {
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
                        // finding the tgt node
                        for node in nodes.iter() {
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
                        let e8 = Edge {
                            src: wff_src_tgt[0].clone(),
                            tgt: wff_src_tgt[1].clone(),
                            e_type: String::from("Wire"),
                            prop: None,
                        };
                        egdes.push(e8);
                    }
                }
            }
            _ => {} // for now only expression in the module level. starting "easy"
        }
        start += 1;
        bf_counter += 1;
    }

    // add wires for inbetween attribute level boxes, so opo's and opi's between attributes get wired through module level wff field, will require reading through node list to match contents field to box field on wff entries
    // assume pof and pif order matches directly to opo and opi order in sub modules
    if !gromet.r#fn.wff.as_ref().is_none() {
        for wire in gromet.r#fn.wff.as_ref().unwrap().iter() {
            let src_idx = wire.src;
            let tgt_idx = wire.tgt;
            let src_id = gromet.r#fn.pif.as_ref().unwrap()[(src_idx - 1) as usize]
                .id
                .unwrap();
            let src_box = gromet.r#fn.pif.as_ref().unwrap()[(src_idx - 1) as usize].r#box;
            let src_contents = gromet.r#fn.bf.as_ref().unwrap()[(src_box - 1) as usize]
                .contents
                .unwrap();
            let tgt_id = gromet.r#fn.pof.as_ref().unwrap()[(tgt_idx - 1) as usize]
                .id
                .unwrap();
            let tgt_box = gromet.r#fn.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].r#box;
            let tgt_contents = gromet.r#fn.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .contents
                .unwrap();

            let mut wff_src_tgt: Vec<String> = vec![];
            // find the src
            for node in nodes.iter() {
                if node.contents == src_contents {
                    if node.n_type == "Opi" {
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src
                            if (src_id as u32) == *p {
                                wff_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
            // find the tgt
            for node in nodes.iter() {
                if node.contents == tgt_contents {
                    if node.n_type == "Opo" {
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src
                            if (tgt_id as u32) == *p {
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
            egdes.push(e9);
        }
    }
    // convert every node object into a node query
    let create = String::from("CREATE");
    for node in nodes.iter() {
        let mut name = String::from("a");
        let mut value = String::from("b");

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
    for edge in egdes.iter() {
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

pub fn parse_gromet_queries(gromet: Gromet) -> Vec<String> {
    let mut queries: Vec<String> = vec![];

    let mut start: u32 = 0;

    queries.append(&mut create_metadata_node(&gromet));
    queries.append(&mut create_module(&gromet));
    queries.append(&mut create_function_net(&gromet, start));

    return queries;
}
