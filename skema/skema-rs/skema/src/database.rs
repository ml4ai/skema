//! Interface to the graph database we are using for persisting GroMEt objects and performing
//! queries on them. We currently use MemgraphDB, an in-memory graph database.

//!! Currently the literals in the opening main function are getting double wired into the + primitive, guessing some external wiring bug...

// wfopi: pif -> opi
// wff: pif -> pof
// wfopo: opo -> pof
// wopio: opo -> opi

/* TODO (1/8/23):
-- Update to newest GroMEt spec
-- Refactor repeated function call implementation to be more robust and in line with other methods
-- Check and debug the wiring for complicated edge cases, some of which are in CHIME_SIR
*/

/* Currently I don't think there is support for:
There being a second function call of the same function which contains an expression,
    I believe the wiring will be messed up going into the expression from the second function
*/

use rsmgclient::{ConnectParams, Connection, MgError};
use crate::FunctionType;
use crate::{Attribute, FnType::Import, FunctionNet, GrometBox};
use crate::{
    Files, Grounding, ModuleCollection, Provenance, TextExtraction, ValueMeta, FnType
};

#[derive(Debug, Clone)]
pub struct MetadataNode {
    pub n_type: String,
    pub node_id: String,
    pub metadata_idx: u32,
    pub metadata_type: Vec<Option<String>>,
    pub gromet_version: Vec<Option<String>>,
    pub text_extraction: Vec<Option<TextExtraction>>,
    pub variable_identifier: Vec<Option<String>>,
    pub variable_definition: Vec<Option<String>>,
    pub value: Vec<Option<ValueMeta>>,
    pub grounding: Vec<Option<Vec<Grounding>>>,
    pub name: Vec<Option<String>>,
    pub global_reference_id: Vec<Option<String>>,
    pub files: Vec<Option<Vec<Files>>>,
    pub source_language: Vec<Option<String>>,
    pub source_language_version: Vec<Option<String>>,
    pub data_type: Vec<Option<String>>,
    pub code_file_reference_uid: Vec<Option<String>>,
    pub line_begin: Vec<Option<u32>>,
    pub line_end: Vec<Option<u32>>,
    pub col_begin: Vec<Option<u32>>,
    pub col_end: Vec<Option<u32>>,
    pub provenance: Vec<Option<Provenance>>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub n_type: String,
    pub value: Option<String>,
    pub name: Option<String>,
    pub node_id: String,
    pub out_idx: Option<Vec<u32>>, // opo or pof index, directly matching the wire src/tgt notation
    pub in_indx: Option<Vec<u32>>, // opi or pif index
    pub contents: u32, // This indexes which index this node has inside the attribute list, but for functions this is overloaded
    pub nbox: u8,
    pub att_bf_idx: u32,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub src: String,
    pub tgt: String,
    pub e_type: String,
    pub prop: Option<u32>, // option because of opo's and opi's
}

pub fn execute_query(query: &str, host: &str) -> Result<(), MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        host: Some(host.to_string()),
        ..Default::default()
    };
    let mut connection = Connection::connect(&connect_params)?;

    // Create simple graph.
    connection.execute_without_results(query)?;

    connection.commit()?;

    Ok(())
}
// this will create a deserialized metadata node
fn create_metadata_node(gromet: &ModuleCollection, metadata_idx: u32) -> Vec<MetadataNode> {
    // grabs the deserialized metadata
    let _metadata = gromet.modules[0].metadata_collection.as_ref().unwrap()
        [(metadata_idx.clone() - 1) as usize][0]
        .clone();
    let mut metas: Vec<MetadataNode> = vec![];

    // since there can be an array of metadata after alignment
    let mut metadata_type_vec: Vec<Option<String>> = vec![];
    let mut gromet_version_vec: Vec<Option<String>> = vec![];
    let mut text_extraction_vec: Vec<Option<TextExtraction>> = vec![];
    let mut variable_identifier_vec: Vec<Option<String>> = vec![];
    let mut variable_definition_vec: Vec<Option<String>> = vec![];
    let mut value_vec: Vec<Option<ValueMeta>> = vec![];
    let mut grounding_vec: Vec<Option<Vec<Grounding>>> = vec![];
    let mut name_vec: Vec<Option<String>> = vec![];
    let mut global_reference_id_vec: Vec<Option<String>> = vec![];
    let mut files_vec: Vec<Option<Vec<Files>>> = vec![];
    let mut source_language_vec: Vec<Option<String>> = vec![];
    let mut source_language_version_vec: Vec<Option<String>> = vec![];
    let mut data_type_vec: Vec<Option<String>> = vec![];
    let mut code_file_reference_uid_vec: Vec<Option<String>> = vec![];
    let mut line_begin_vec: Vec<Option<u32>> = vec![];
    let mut line_end_vec: Vec<Option<u32>> = vec![];
    let mut col_begin_vec: Vec<Option<u32>> = vec![];
    let mut col_end_vec: Vec<Option<u32>> = vec![];
    let mut provenance_vec: Vec<Option<Provenance>> = vec![];

    // fill out metadata arrays
    for data in gromet.modules[0].metadata_collection.as_ref().unwrap()[(metadata_idx.clone() - 1) as usize].clone() {
        metadata_type_vec.push(data.metadata_type.clone());
        gromet_version_vec.push(data.gromet_version.clone());
        text_extraction_vec.push(data.text_extraction.clone());
        variable_identifier_vec.push(data.variable_identifier.clone());
        variable_definition_vec.push(data.variable_definition.clone());
        value_vec.push(data.value.clone());
        grounding_vec.push(data.grounding.clone());
        name_vec.push(data.name.clone());
        global_reference_id_vec.push(data.global_reference_id.clone());
        files_vec.push(data.files.clone());
        source_language_vec.push(data.source_language.clone());
        source_language_version_vec.push(data.source_language_version.clone());
        data_type_vec.push(data.data_type.clone());
        code_file_reference_uid_vec.push(data.code_file_reference_uid.clone());
        line_begin_vec.push(data.line_begin.clone());
        line_end_vec.push(data.line_end.clone());
        col_begin_vec.push(data.col_begin.clone());
        col_end_vec.push(data.col_end.clone());
        provenance_vec.push(data.provenance.clone());
    }

    let m1 = MetadataNode {
        n_type: String::from("Metadata"),
        node_id: format!("m{}", metadata_idx),
        metadata_idx: metadata_idx.clone(),
        metadata_type: metadata_type_vec.clone(),
        gromet_version: gromet_version_vec.clone(),
        text_extraction: text_extraction_vec.clone(),
        variable_identifier: variable_identifier_vec.clone(),
        variable_definition: variable_definition_vec.clone(),
        value: value_vec.clone(),
        grounding: grounding_vec.clone(),
        name: name_vec.clone(),
        global_reference_id: global_reference_id_vec.clone(),
        files: files_vec.clone(),
        source_language: source_language_vec.clone(),
        source_language_version: source_language_version_vec.clone(),
        data_type: data_type_vec.clone(),
        code_file_reference_uid: code_file_reference_uid_vec.clone(),
        line_begin: line_begin_vec.clone(),
        line_end: line_end_vec.clone(),
        col_begin: col_begin_vec.clone(),
        col_end: col_end_vec.clone(),
        provenance: provenance_vec.clone(),
    };
    metas.push(m1);
    return metas;
}
// creates the metadata node query
fn create_metadata_node_query(meta_node: MetadataNode) -> Vec<String> {

    // determine vec length
    let metadata_len = meta_node.gromet_version.len();

    // construct the metadata fields
    let mut metadata_type_q: Vec<String> = vec![];
    let mut gromet_version_q: Vec<String> = vec![];
    let mut text_extraction_q: Vec<String> = vec![];
    let mut variable_identifier_q: Vec<String> = vec![];
    let mut variable_definition_q: Vec<String> = vec![];
    let mut value_q: Vec<String> = vec![];
    let mut grounding_q: Vec<String> = vec![];
    let mut name_q: Vec<String> = vec![];
    let mut global_reference_id_q: Vec<String> = vec![];
    let mut files_q: Vec<String> = vec![];
    let mut source_language_q: Vec<String> = vec![];
    let mut source_language_version_q: Vec<String> = vec![];
    let mut data_type_q: Vec<String> = vec![];
    let mut code_file_reference_uid_q: Vec<String> = vec![];
    let mut line_begin_q: Vec<u32> = vec![];
    let mut line_end_q: Vec<u32> = vec![];
    let mut col_begin_q: Vec<u32> = vec![];
    let mut col_end_q: Vec<u32> = vec![];
    let mut provenance_q: Vec<String> = vec![];

    for i in 0..metadata_len {
        metadata_type_q.push(meta_node.metadata_type[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        gromet_version_q.push(meta_node.gromet_version[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        text_extraction_q.push(meta_node.text_extraction[i].as_ref().map_or_else(|| String::from(""),|x| format!("{:?}", x)));
        variable_identifier_q.push(meta_node.variable_identifier[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        variable_definition_q.push(meta_node.variable_definition[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        value_q.push(meta_node.value[i].as_ref().map_or_else(|| String::from(""),|x| format!("{:?}", x)));
        grounding_q.push(meta_node.grounding[i].as_ref().map_or_else(|| String::from(""),|x| format!("{:?}", x)));
        name_q.push(meta_node.name[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        global_reference_id_q.push(meta_node.global_reference_id[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        files_q.push(meta_node.files[i].as_ref().map_or_else(|| String::from(""),|x| format!("{:?}", x)));
        source_language_q.push(meta_node.source_language[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        source_language_version_q.push(meta_node.source_language_version[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        data_type_q.push(meta_node.data_type[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        code_file_reference_uid_q.push(meta_node.code_file_reference_uid[i].as_ref().map_or_else(|| String::from(""),|x| format!("{}", x)));
        line_begin_q.push(meta_node.line_begin[i].as_ref().map_or_else(|| 0,|x| *x));
        line_end_q.push(meta_node.line_end[i].as_ref().map_or_else(|| 0,|x| *x));
        col_begin_q.push(meta_node.col_begin[i].as_ref().map_or_else(|| 0,|x| *x));
        col_end_q.push(meta_node.col_end[i].as_ref().map_or_else(|| 0,|x| *x));
        provenance_q.push(meta_node.provenance[i].as_ref().map_or_else(|| String::from(""),|x| format!("{:?}", x)));
    }

    // construct the queries
    let mut queries: Vec<String> = vec![];
    let create = String::from("CREATE");
    let metanode_query = format!(
        "{} ({}:{} {{metadata_idx:{:?},metadata_type:{:?},gromet_version:{:?},text_extraction:{:?},variable_identifier:{:?},variable_definition:{:?},value:{:?},grounding:{:?},name:{:?},global_reference_id:{:?},files:{:?},source_language:{:?}
            ,source_language_version:{:?},data_type:{:?},code_file_reference_uid:{:?},line_begin:{:?},line_end:{:?}
            ,col_begin:{:?},col_end:{:?},provenance:{:?}}})",
        create, meta_node.node_id, meta_node.n_type, meta_node.metadata_idx, 
        metadata_type_q,
        gromet_version_q,
        text_extraction_q,
        variable_identifier_q,
        variable_definition_q,
        value_q,
        grounding_q,
        name_q,
        global_reference_id_q,
        files_q,
        source_language_q,
        source_language_version_q,
        data_type_q,
        code_file_reference_uid_q,
        line_begin_q,
        line_end_q,
        col_begin_q,
        col_end_q,
        provenance_q
    );
    queries.push(metanode_query);
    return queries;
}
fn create_module(gromet: &ModuleCollection) -> Vec<String> {
    let mut queries: Vec<String> = vec![];

    let create = String::from("CREATE");

    let node_label = String::from("mod:Module");

    let schema = format!("schema:{:?}", gromet.modules[0].schema);
    let schema_version = format!("schema_version:{:?}", gromet.modules[0].schema_version);
    let filename = format!("filename:{:?}", gromet.modules[0].name);
    let name = format!(
        "name:{:?}",
        gromet.modules[0].r#fn.b.as_ref().unwrap()[0]
            .name
            .as_ref()
            .unwrap()
    );
    let metadata_idx = gromet.modules[0].r#fn.b.as_ref().unwrap()[0]
        .metadata
        .as_ref()
        .unwrap();

    let node_query = format!(
        "{} ({} {{{},{},{},{}}})",
        create, node_label, schema, schema_version, filename, name
    );
    queries.push(node_query);

    let meta = create_metadata_node(&gromet, metadata_idx.clone());
    let mut meta_query: Vec<String> = vec![];
    for node in meta.iter() {
        meta_query.append(&mut create_metadata_node_query(node.clone()))
    }
    queries.append(&mut meta_query);

    let me1 = Edge {
        src: String::from("mod"),
        tgt: format!("m{}", metadata_idx),
        e_type: String::from("Metadata"),
        prop: None,
    };
    let edge_query = format!(
        "{} ({})-[e{}{}:{}]->({})",
        create, me1.src, me1.src, me1.tgt, me1.e_type, me1.tgt
    );
    queries.push(edge_query);

    return queries;
}

fn create_graph_queries(gromet: &ModuleCollection, start: u32) -> Vec<String> {
    let mut queries: Vec<String> = vec![];
    // if a library module need to walk through gromet differently
    if gromet.modules[0].r#fn.bf.is_none() {
        queries.append(&mut create_function_net_lib(&gromet, start));
    } else {
        // if executable code
        queries.append(&mut create_function_net(&gromet, start));
    }
    return queries;
}

// This creates the graph queries from a function network if the code is not executable
// currently only supports creating the first attribute (a function) and all its dependencies
// need to add support to find next function and create network as well and repeat
#[allow(unused_assignments)]
fn create_function_net_lib(gromet: &ModuleCollection, mut start: u32) -> Vec<String> {
    let mut queries: Vec<String> = vec![];
    let mut nodes: Vec<Node> = vec![];
    let mut meta_nodes: Vec<MetadataNode> = vec![];
    let mut metadata_idx = 0;
    let mut edges: Vec<Edge> = vec![];
    // in order to have less repetition for multiple function calls and to setup support for recursive functions
    // We check if the function node and thus contents were already made, and not duplicate the contents if already made
    let mut bf_counter: u8 = 1;
    let mut function_call_repeat = false;
    let mut original_bf = bf_counter.clone();
    let _boxf = gromet.modules[0].attributes[0].value.clone();
    for node in nodes.clone() {
        if (1 == node.contents) && (node.n_type == "Function") {
            function_call_repeat = true;
            if node.nbox < original_bf {
                original_bf = node.nbox.clone(); // This grabs the first instance of bf that the function was called
                                                 // and thus is the nbox value of the nodes of the original contents
            }
        }
    }
    if function_call_repeat {
        // This means the function has been called before so we don't fully construct the graph
        // constructing metadata node if metadata exists
        let mut metadata_idx = 0;
        let eboxf = gromet.modules[0].attributes[0 as usize].clone();
        let n1 = Node {
            n_type: String::from("Function"),
            value: None,
            name: Some(
                eboxf.value.b.as_ref().unwrap()[0]
                    .name
                    .clone()
                    .map_or_else(|| format!("Function{}", start), |x| format!("{}", x)),
            ),
            node_id: format!("n{}", start),
            out_idx: None,
            in_indx: None,
            contents: 1,
            nbox: bf_counter,
            att_bf_idx: 0,
        };
        let e1 = Edge {
            src: String::from("mod"),
            tgt: format!("n{}", start),
            e_type: String::from("Contains"),
            prop: Some(1),
        };
        nodes.push(n1.clone());
        edges.push(e1);
        if !eboxf.value.b.as_ref().unwrap()[0]
            .metadata
            .as_ref()
            .is_none()
        {
            metadata_idx = eboxf.value.b.unwrap()[0].metadata.unwrap().clone();
            let mut repeat_meta = false;
            for node in meta_nodes.iter() {
                if node.metadata_idx == metadata_idx {
                    repeat_meta = true;
                }
            }
            if !repeat_meta {
                meta_nodes.append(&mut create_metadata_node(
                    &gromet.clone(),
                    metadata_idx.clone(),
                ));
                // adding the metadata edge
                let me1 = Edge {
                    src: n1.node_id.clone(),
                    tgt: format!("m{}", metadata_idx),
                    e_type: String::from("Metadata"),
                    prop: None,
                };
                edges.push(me1);
            }
        }
        // we still construct unique ports for this function, however the contents will not be repeated
        start += 1;
        let idx = 0;
        let eboxf = gromet.modules[0].attributes[idx as usize].clone();
        // create the ports
        (nodes, edges, meta_nodes, start) = create_opo(
            nodes.clone(),
            edges.clone(),
            meta_nodes.clone(),
            &gromet.clone(),
            start.clone(),
            n1.clone(),
            bf_counter.clone(),
        );
        (nodes, edges, meta_nodes, start) = create_opi(
            nodes.clone(),
            edges.clone(),
            meta_nodes.clone(),
            &gromet.clone(),
            start.clone(),
            n1.clone(),
            bf_counter.clone(),
        );
        // now to add the contains wires for the additional function call onto the original contents nodes:
        for node in nodes.clone() {
            if (node.nbox == original_bf) && (node.contents == (idx + 1)) {
                if (node.n_type == "Literal")
                    || (node.n_type == "Primitive")
                    || (node.n_type == "Predicate")
                    || (node.n_type == "Expression")
                {
                    let e5 = Edge {
                        src: n1.node_id.clone(),
                        tgt: node.node_id.clone(),
                        e_type: String::from("Contains"),
                        prop: None,
                    };
                    edges.push(e5);
                }
            }
        }

        // now we need to wire these ports to the content nodes which already exist.
        // they will have the same contents, being: (idx+1), however the bf_counter will be different, parse bf_counter from first call
        // (smallest of bf_counter of all calls) and use that in wiring, it is original_bf now
        // concerns over wiring into an expression, the expression would be in the correct contents attribute, but the ports are labeled as the expressions contents
        for wire in eboxf.value.wfopi.unwrap().iter() {
            let mut wfopi_src_tgt: Vec<String> = vec![];
            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if original_bf == node.nbox {
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
                if original_bf == node.nbox {
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
    } else {
        /////////////////////////
        // This is for no function repeat in a library representation
        /////////////////////////
        let eboxf = gromet.modules[0].attributes[0 as usize].clone();
        let n1 = Node {
            n_type: String::from("Function"),
            value: None,
            name: Some(
                eboxf.value.b.as_ref().unwrap()[0]
                    .name
                    .clone()
                    .map_or_else(|| format!("Function{}", start), |x| format!("{}", x)),
            ),
            node_id: format!("n{}", start),
            out_idx: None,
            in_indx: None,
            contents: 1,
            nbox: bf_counter,
            att_bf_idx: 0,
        };
        let e1 = Edge {
            src: String::from("mod"),
            tgt: format!("n{}", start),
            e_type: String::from("Contains"),
            prop: Some(1),
        };
        nodes.push(n1.clone());
        edges.push(e1);

        if !eboxf.value.b.as_ref().unwrap()[0]
            .metadata
            .as_ref()
            .is_none()
        {
            metadata_idx = eboxf.value.b.unwrap()[0].metadata.unwrap().clone();
            let mut repeat_meta = false;
            for node in meta_nodes.iter() {
                if node.metadata_idx == metadata_idx {
                    repeat_meta = true;
                }
            }
            if !repeat_meta {
                meta_nodes.append(&mut create_metadata_node(
                    &gromet.clone(),
                    metadata_idx.clone(),
                ));
                // adding the metadata edge
                let me1 = Edge {
                    src: n1.node_id.clone(),
                    tgt: format!("m{}", metadata_idx),
                    e_type: String::from("Metadata"),
                    prop: None,
                };
                edges.push(me1);
            }
        }
        // now travel to contents index of the attribute list (note it is 1 index,
        // so contents=1 => attribute[0])
        // create nodes and edges for this entry, include opo's and opi's
        start += 1;
        let idx = 1;
        let eboxf = gromet.modules[0].attributes[(idx - 1) as usize].clone();
        // create the ports
        (nodes, edges, meta_nodes, start) = create_opo(
            nodes.clone(),
            edges.clone(),
            meta_nodes.clone(),
            &gromet.clone(),
            start.clone(),
            n1.clone(),
            bf_counter.clone(),
        );
        (nodes, edges, meta_nodes, start) = create_opi(
            nodes.clone(),
            edges.clone(),
            meta_nodes.clone(),
            &gromet.clone(),
            start.clone(),
            n1.clone(),
            bf_counter.clone(),
        );
        // now to construct the nodes inside the function, currently supported Literals and Primitives
        // first include an Expression for increased depth
        let mut box_counter: u8 = 1;
        for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
            match sboxf.function_type {
                FunctionType::Predicate => {
                    (nodes, edges, start, meta_nodes) = create_att_predicate(
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
                        meta_nodes.clone(),
                    );
                }
                FunctionType::Expression => {
                    (nodes, edges, start, meta_nodes) = create_att_expression(
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
                        meta_nodes.clone(),
                    );
                }
                FunctionType::Literal => {
                    (nodes, edges, meta_nodes) = create_att_literal(
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
                        meta_nodes.clone(),
                    );
                }
                FunctionType::Primitive => {
                    (nodes, edges, meta_nodes) = create_att_primitive(
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
                        meta_nodes.clone(),
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
            idx.clone() ,
            bf_counter.clone(),
        );
        // perform cross attributal wiring of function
        (edges, nodes) = cross_att_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone() ,
            bf_counter.clone(),
        );
    }
    start += 1;
    bf_counter += 1;

    // add wires for inbetween attribute level boxes, so opo's, opi's and module level literals
    // between attributes
    // get wired through module level wff field, will require reading through node list to
    // match contents field to box field on wff entries
    edges = external_wiring(&gromet, nodes.clone(), edges);

    // make conditionals if they exist
    if !gromet.modules[0].r#fn.bc.as_ref().is_none() {
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
            att_bf_idx: 0,
        };
        for _cond in gromet.modules[0].r#fn.bc.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            (nodes, edges, start, meta_nodes) = create_conditional(
                &gromet.clone(),
                gromet.modules[0].r#fn.clone(), // This is gromet but is more generalizable based on scope
                nodes.clone(),
                edges.clone(),
                temp_mod_node.clone(),
                0,            // because top level
                cond_counter, // This indexes the conditional in the list of conditionals (bc)
                0,            // because top level
                start.clone(),
                meta_nodes.clone(),
            );
            cond_counter += 1;
        }
    }
    // make conditionals if they exist
    if !gromet.modules[0].r#fn.bl.as_ref().is_none() {
        let mut while_counter = 0;
        let temp_mod_node = Node {
            n_type: String::from("module"),
            value: None,
            name: None, // I think this naming will get messed up if there are multiple ports...
            node_id: format!("mod"),
            out_idx: None,
            in_indx: None,
            contents: 0,
            nbox: 0,
            att_bf_idx: 0,
        };
        for _while_l in gromet.modules[0].r#fn.bl.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            (nodes, edges, start, meta_nodes) = create_while_loop(
                &gromet.clone(),
                gromet.modules[0].r#fn.clone(), // This is gromet but is more generalizable based on scope
                nodes.clone(),
                edges.clone(),
                temp_mod_node.clone(),
                0,             // because top level
                while_counter, // This indexes the conditional in the list of conditionals (bc)
                0,             // because top level
                start.clone(),
                meta_nodes.clone(),
            );
            while_counter += 1;
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
            value = String::from("");
        } else {
            value = format!("{}", node.value.as_ref().unwrap());
        }
        let node_query = format!(
            "{} ({}:{} {{name:{:?},value:{:?},order_box:{:?},order_att:{:?}}})",
            create, node.node_id, node.n_type, name, value, node.nbox, node.contents
        );
        queries.push(node_query);
    }
    for node in meta_nodes.iter() {
        queries.append(&mut create_metadata_node_query(node.clone()));
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

#[allow(unused_assignments)]
fn create_function_net(gromet: &ModuleCollection, mut start: u32) -> Vec<String> {
    // intialize the vectors
    let mut queries: Vec<String> = vec![];
    let mut nodes: Vec<Node> = vec![];
    let mut meta_nodes: Vec<MetadataNode> = vec![];
    let mut metadata_idx = 0;
    let mut edges: Vec<Edge> = vec![];

    let mut bf_counter: u8 = 1;
    for boxf in gromet.modules[0].r#fn.bf.as_ref().unwrap().iter() {
        // construct the sub module level boxes along with their metadata and connection to module
        match boxf.function_type {
            FunctionType::Literal => {
                // first we find the pof value for the literal box
                let mut pof: Vec<u32> = vec![];
                if !gromet.modules[0].r#fn.pof.clone().is_none() {
                    let mut po_idx: u32 = 1;
                    for port in gromet.modules[0].r#fn.pof.clone().unwrap().iter() {
                        if port.r#box == bf_counter {
                            pof.push(po_idx);
                        }
                        po_idx += 1;
                    }
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
                    att_bf_idx: 0,
                };
                let e1 = Edge {
                    src: String::from("mod"),
                    tgt: format!("n{}", start),
                    e_type: String::from("Contains"),
                    prop: boxf.contents,
                };
                nodes.push(n1.clone());
                edges.push(e1);
                if !boxf.metadata.as_ref().is_none() {
                    metadata_idx = boxf.metadata.unwrap();
                    let mut repeat_meta = false;
                    for node in meta_nodes.iter() {
                        if node.metadata_idx == metadata_idx {
                            repeat_meta = true;
                        }
                    }
                    if !repeat_meta {
                        meta_nodes.append(&mut create_metadata_node(
                            &gromet.clone(),
                            metadata_idx.clone(),
                        ));
                        let me1 = Edge {
                            src: n1.node_id.clone(),
                            tgt: format!("m{}", metadata_idx),
                            e_type: String::from("Metadata"),
                            prop: None,
                        };
                        edges.push(me1);
                    }
                }
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
                    att_bf_idx: 0,
                };
                let e1 = Edge {
                    src: String::from("mod"),
                    tgt: format!("n{}", start),
                    e_type: String::from("Contains"),
                    prop: boxf.contents,
                };
                nodes.push(n1.clone());
                edges.push(e1);

                if !boxf.metadata.as_ref().is_none() {
                    metadata_idx = boxf.metadata.unwrap();
                    let mut repeat_meta = false;
                    for node in meta_nodes.iter() {
                        if node.metadata_idx == metadata_idx {
                            repeat_meta = true;
                        }
                    }
                    if !repeat_meta {
                        meta_nodes.append(&mut create_metadata_node(
                            &gromet.clone(),
                            metadata_idx.clone(),
                        ));
                        let me1 = Edge {
                            src: n1.node_id.clone(),
                            tgt: format!("m{}", metadata_idx),
                            e_type: String::from("Metadata"),
                            prop: None,
                        };
                        edges.push(me1);
                    }
                }

                // now travel to contents index of the attribute list (note it is 1 index,
                // so contents=1 => attribute[0])
                // create nodes and edges for this entry, include opo's and opi's
                start += 1;
                let idx = boxf.contents.unwrap() - 1;
                let eboxf = gromet.modules[0].attributes[idx as usize].clone();
                // create the ports
                (nodes, edges, meta_nodes, start) = create_opo(
                    nodes.clone(),
                    edges.clone(),
                    meta_nodes.clone(),
                    &gromet.clone(),
                    start.clone(),
                    n1.clone(),
                    bf_counter.clone(),
                );
                (nodes, edges, meta_nodes, start) = create_opi(
                    nodes.clone(),
                    edges.clone(),
                    meta_nodes.clone(),
                    &gromet.clone(),
                    start.clone(),
                    n1.clone(),
                    bf_counter.clone(),
                );
                // now to construct the nodes inside the expression, Literal and Primitives
                let mut box_counter: u8 = 1;
                for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
                    match sboxf.function_type {
                        FunctionType::Literal => {
                            (nodes, edges, meta_nodes) = create_att_literal(
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
                                meta_nodes.clone(),
                            );
                        }
                        FunctionType::Primitive => {
                            (nodes, edges, meta_nodes) = create_att_primitive(
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
                                meta_nodes.clone(),
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
                    att_bf_idx: 0,
                };
                let e1 = Edge {
                    src: String::from("mod"),
                    tgt: format!("n{}", start),
                    e_type: String::from("Contains"),
                    prop: boxf.contents,
                };
                nodes.push(n1.clone());
                edges.push(e1);
                if !boxf.metadata.as_ref().is_none() {
                    metadata_idx = boxf.metadata.unwrap();
                    let mut repeat_meta = false;
                    for node in meta_nodes.iter() {
                        if node.metadata_idx == metadata_idx {
                            repeat_meta = true;
                        }
                    }
                    if !repeat_meta {
                        meta_nodes.append(&mut create_metadata_node(
                            &gromet.clone(),
                            metadata_idx.clone(),
                        ));
                        let me1 = Edge {
                            src: n1.node_id.clone(),
                            tgt: format!("m{}", metadata_idx),
                            e_type: String::from("Metadata"),
                            prop: None,
                        };
                        edges.push(me1);
                    }
                }

                // now travel to contents index of the attribute list (note it is 1 index,
                // so contents=1 => attribute[0])
                // create nodes and edges for this entry, include opo's and opi's
                start += 1;
                let idx = boxf.contents.unwrap() - 1;

                let eboxf = gromet.modules[0].attributes[idx as usize].clone();
                (nodes, edges, meta_nodes, start) = create_opo(
                    nodes.clone(),
                    edges.clone(),
                    meta_nodes.clone(),
                    &gromet.clone(),
                    start.clone(),
                    n1.clone(),
                    bf_counter.clone(),
                );
                (nodes, edges, meta_nodes, start) = create_opi(
                    nodes.clone(),
                    edges.clone(),
                    meta_nodes.clone(),
                    &gromet.clone(),
                    start.clone(),
                    n1.clone(),
                    bf_counter.clone(),
                );
                // now to construct the nodes inside the expression, Literal and Primitives
                let mut box_counter: u8 = 1;
                for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
                    match sboxf.function_type {
                        FunctionType::Literal => {
                            (nodes, edges, meta_nodes) = create_att_literal(
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
                                meta_nodes.clone(),
                            );
                        }
                        FunctionType::Primitive => {
                            (nodes, edges, meta_nodes) = create_att_primitive(
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
                                meta_nodes.clone(),
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
                // in order to have less repetition for multiple function calls and to setup support for recursive functions
                // We check if the function node and thus contents were already made, and not duplicate the contents if already made
                let mut function_call_repeat = false;
                let mut original_bf = bf_counter.clone();
                for node in nodes.clone() {
                    if (boxf.contents.unwrap() == node.contents) && (node.n_type == "Function") {
                        function_call_repeat = true;
                        if node.nbox < original_bf {
                            original_bf = node.nbox.clone(); // This grabs the first instance of bf that the function was called
                                                             // and thus is the nbox value of the nodes of the original contents
                        }
                    }
                }
                if function_call_repeat {
                    // This means the function has been called before so we don't fully construct the graph
                    // still construct the function call node and its metadata and contains edge

                    // functions will have a name and additional metadata coming from the "b" field
                    let idx = boxf.contents.unwrap() - 1;
                    let eboxf = gromet.modules[0].attributes[idx as usize].clone();

                    let n1 = Node {
                        n_type: String::from("Function"),
                        value: None,
                        name: Some(
                            eboxf.value.b.as_ref().unwrap()[0]
                                .name
                                .clone()
                                .map_or_else(|| format!("Function{}", start), |x| format!("{}", x)),
                        ),
                        node_id: format!("n{}", start),
                        out_idx: None,
                        in_indx: None,
                        contents: idx + 1,
                        nbox: bf_counter,
                        att_bf_idx: 0,
                    };
                    let e1 = Edge {
                        src: String::from("mod"),
                        tgt: format!("n{}", start),
                        e_type: String::from("Contains"),
                        prop: boxf.contents,
                    };
                    nodes.push(n1.clone());
                    edges.push(e1);
                    // bf level metadata reference
                    if !boxf.metadata.as_ref().is_none() {
                        metadata_idx = boxf.metadata.unwrap();
                        let mut repeat_meta = false;
                        for node in meta_nodes.iter() {
                            if node.metadata_idx == metadata_idx {
                                repeat_meta = true;
                            }
                        }
                        if !repeat_meta {
                            meta_nodes.append(&mut create_metadata_node(
                                &gromet.clone(),
                                metadata_idx.clone(),
                            ));
                            let me1 = Edge {
                                src: n1.node_id.clone(),
                                tgt: format!("m{}", metadata_idx),
                                e_type: String::from("Metadata"),
                                prop: None,
                            };
                            edges.push(me1);
                        }
                    }
                    // attribute b level metadata reference
                    if !eboxf.value.b.as_ref().unwrap()[0]
                        .metadata
                        .as_ref()
                        .is_none()
                    {
                        metadata_idx = eboxf.value.b.unwrap()[0].metadata.unwrap().clone();
                        let mut repeat_meta = false;
                        for node in meta_nodes.iter() {
                            if node.metadata_idx == metadata_idx {
                                repeat_meta = true;
                            }
                        }
                        if !repeat_meta {
                            meta_nodes.append(&mut create_metadata_node(
                                &gromet.clone(),
                                metadata_idx.clone(),
                            ));
                            let me1 = Edge {
                                src: n1.node_id.clone(),
                                tgt: format!("m{}", metadata_idx),
                                e_type: String::from("Metadata"),
                                prop: None,
                            };
                            edges.push(me1);
                        }
                    }
                    // we still construct unique ports for this function, however the contents will not be repeated
                    start += 1;
                    // construct opo nodes, if not none
                    (nodes, edges, meta_nodes, start) = create_opo(
                        nodes.clone(),
                        edges.clone(),
                        meta_nodes.clone(),
                        &gromet.clone(),
                        start.clone(),
                        n1.clone(),
                        bf_counter.clone(),
                    );
                    (nodes, edges, meta_nodes, start) = create_opi(
                        nodes.clone(),
                        edges.clone(),
                        meta_nodes.clone(),
                        &gromet.clone(),
                        start.clone(),
                        n1.clone(),
                        bf_counter.clone(),
                    );
                    // now to add the contains wires for the additional function call onto the original contents nodes:
                    for node in nodes.clone() {
                        if (node.nbox == original_bf) && (node.contents == (idx + 1)) {
                            if (node.n_type == "Literal")
                                || (node.n_type == "Primitive")
                                || (node.n_type == "Predicate")
                                || (node.n_type == "Expression")
                            {
                                let e5 = Edge {
                                    src: n1.node_id.clone(),
                                    tgt: node.node_id.clone(),
                                    e_type: String::from("Contains"),
                                    prop: None,
                                };
                                edges.push(e5);
                            }
                        }
                    }

                    // now we need to wire these ports to the content nodes which already exist.
                    // they will have the same contents, being: (idx+1), however the bf_counter will be different, parse bf_counter from first call
                    // (smallest of bf_counter of all calls) and use that in wiring, it is original_bf now
                    // concerns over wiring into an expression, the expression would be in the correct contents attribute, but the ports are labeled as the expressions contents
                    for wire in eboxf.value.wfopi.unwrap().iter() {
                        let mut wfopi_src_tgt: Vec<String> = vec![];
                        // find the src node
                        for node in nodes.iter() {
                            // make sure in correct box
                            if original_bf == node.nbox {
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
                            if original_bf == node.nbox {
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
                } else {
                    // get attribute function net to pass into function
                    let pfn = gromet.modules[0].attributes[(boxf.contents.unwrap() - 1) as usize]
                        .value
                        .clone();

                    // need a parent node for the module at this level
                    let temp_mod_node = Node {
                        n_type: String::from("module"),
                        value: None,
                        name: None,
                        node_id: format!("mod"),
                        out_idx: None,
                        in_indx: None,
                        contents: 0,
                        nbox: 0,
                        att_bf_idx: 0,
                    };

                    (nodes, edges, start, meta_nodes) = create_function(
                        &gromet.clone(),                // gromet for metadata
                        pfn.clone(),                    // function net for attribute
                        nodes.clone(),                  // nodes
                        edges.clone(),                  // edges
                        temp_mod_node.clone(),          // parent node
                        boxf.contents.unwrap().clone(), // attribute idx
                        bf_counter.clone(),             // top level bf idx
                        0,                              // sub box idx, 0 for top level
                        start.clone(),                  // start
                        meta_nodes.clone(),             // metadata nodes
                    );
                }
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

    // make conditionals if they exist
    if !gromet.modules[0].r#fn.bc.as_ref().is_none() {
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
            att_bf_idx: 0,
        };
        for _cond in gromet.modules[0].r#fn.bc.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            (nodes, edges, start, meta_nodes) = create_conditional(
                &gromet.clone(),
                gromet.modules[0].r#fn.clone(), // This is gromet but is more generalizable based on scope
                nodes.clone(),
                edges.clone(),
                temp_mod_node.clone(),
                0,            // because top level
                cond_counter, // This indexes the conditional in the list of conditionals (bc)
                0,            // because top level
                start.clone(),
                meta_nodes.clone(),
            );
            cond_counter += 1;
        }
    }
    // make loops if they exist
    if !gromet.modules[0].r#fn.bl.as_ref().is_none() {
        let mut while_counter = 0;
        let temp_mod_node = Node {
            n_type: String::from("module"),
            value: None,
            name: None, // I think this naming will get messed up if there are multiple ports...
            node_id: format!("mod"),
            out_idx: None,
            in_indx: None,
            contents: 0,
            nbox: 0,
            att_bf_idx: 0,
        };
        for _while_l in gromet.modules[0].r#fn.bl.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            (nodes, edges, start, meta_nodes) = create_while_loop(
                &gromet.clone(),
                gromet.modules[0].r#fn.clone(), // This is gromet but is more generalizable based on scope
                nodes.clone(),
                edges.clone(),
                temp_mod_node.clone(),
                0,             // because top level
                while_counter, // This indexes the conditional in the list of conditionals (bc)
                0,             // because top level
                start.clone(),
                meta_nodes.clone(),
            );
            while_counter += 1;
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
            value = String::from("");
        } else {
            value = format!("{}", node.value.as_ref().unwrap());
        }
        let node_query = format!(
            "{} ({}:{} {{name:{:?},value:{:?},order_box:{:?},order_att:{:?}}})",
            create, node.node_id, node.n_type, name, value, node.nbox, node.contents
        );
        queries.push(node_query);
    }
    for node in meta_nodes.iter() {
        queries.append(&mut create_metadata_node_query(node.clone()));
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
// this method creates an import type function
#[allow(unused_assignments)]
pub fn create_import(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    _function_net: FunctionNet, // This is gromet but is more generalizable based on scope
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node, // used for contains construction
    att_idx: u32,      // This will index the attribute the function is in
    bf_counter: u8, // This indexes which box the function is under, inherited from parent if not explicit
    att_bf_idx: u32, // This indexes if the function is a subscope of a larger function, 0 if not
    start: u32,     // for node and edge indexing
    mut meta_nodes: Vec<MetadataNode>,
) -> (Vec<Node>, Vec<Edge>, u32, Vec<MetadataNode>) {
    let eboxf = gromet.modules[0].clone();
    let sboxf = gromet.modules[0].attributes[(att_idx - 1) as usize]
        .value
        .clone();
    let mboxf = eboxf.r#fn.bf.unwrap()[(bf_counter - 1) as usize].clone();

    let mut pof: Vec<u32> = vec![];
    if !eboxf.r#fn.pof.clone().is_none() {
        let mut po_idx: u32 = 1;
        for port in eboxf.r#fn.pof.clone().unwrap().iter() {
            if port.r#box == bf_counter {
                pof.push(po_idx);
            }
            po_idx += 1;
        }
    }
    // then find pif's for box
    let mut pif: Vec<u32> = vec![];
    if !eboxf.r#fn.pif.clone().is_none() {
        let mut pi_idx: u32 = 1;
        for port in eboxf.r#fn.pif.clone().unwrap().iter() {
            if port.r#box == bf_counter {
                pif.push(pi_idx);
            }
            pi_idx += 1;
        }
    }
    // now make the node with the port information
    let mut metadata_idx = 0;
    let n3 = Node {
        n_type: String::from("Import"),
        value: None,
        name: sboxf.name.clone(),
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: Some(pif),
        contents: att_idx,
        nbox: bf_counter,
        att_bf_idx: att_bf_idx,
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
    if !mboxf.metadata.is_none() {
        metadata_idx = mboxf.metadata.clone().unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(
                &gromet.clone(),
                metadata_idx.clone(),
            ));
            let me1 = Edge {
                src: n3.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }

    return (nodes, edges, start, meta_nodes);
}

// this creates a function node including all the contents included in it, including additional functions
#[allow(unused_assignments)]
pub fn create_function(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    function_net: FunctionNet, // This is gromet but is more generalizable based on scope
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node, // used for contains construction
    att_idx: u32,      // This will index the attribute the function is in
    bf_counter: u8, // This indexes which box the function is under, inherited from parent if not explicit
    att_bf_idx: u32, // This indexes if the function is a subscope of a larger function, 0 if not
    mut start: u32, // for node and edge indexing
    mut meta_nodes: Vec<MetadataNode>,
) -> (Vec<Node>, Vec<Edge>, u32, Vec<MetadataNode>) {
    // function is not repeated
    let eboxf = gromet.modules[0].attributes[(att_idx - 1) as usize].clone();

    // now we add a check for if this is an imported function
    match eboxf.r#type.clone() {
        Import => {
            (nodes, edges, start, meta_nodes) = create_import(
                &gromet.clone(),
                function_net.clone(),
                nodes.clone(),
                edges.clone(),
                parent_node.clone(),
                att_idx.clone(),
                bf_counter.clone(),
                att_bf_idx.clone(),
                start.clone(),
                meta_nodes.clone(),
            );

            // now to implement wiring
            edges = import_wiring(
                &gromet.clone(),
                eboxf.clone(),
                nodes.clone(),
                edges,
                att_idx.clone(),
                bf_counter.clone(),
                parent_node.clone(),
            );

            return (nodes, edges, start, meta_nodes);
        }
        FnType::Fn => {
            let n1 = Node {
                n_type: String::from("Function"),
                value: None,
                name: Some(
                    eboxf.value.b.as_ref().unwrap()[0]
                        .name
                        .clone()
                        .map_or_else(|| format!("Function{}", start), |x| format!("{}", x)),
                ),
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: None,
                contents: att_idx,
                nbox: bf_counter,
                att_bf_idx: att_bf_idx,
            };
            let e1 = Edge {
                src: parent_node.node_id.clone(),
                tgt: n1.node_id.clone(),
                e_type: String::from("Contains"),
                prop: Some(att_idx),
            };
            nodes.push(n1.clone());
            edges.push(e1);
            let mut metadata_idx = 0;
            // attribute b level metadata reference
            if !eboxf.value.b.as_ref().unwrap()[0]
                .metadata
                .as_ref()
                .is_none()
            {
                metadata_idx = eboxf.value.b.as_ref().unwrap()[0].metadata.unwrap().clone();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n1.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            // initial function node has been constructed, based on given inputs

            // now travel to contents index of the attribute list (note it is 1 index,
            // so contents=1 => attribute[0])
            // create nodes and edges for this entry, include opo's and opi's
            start += 1;

            // construct opo nodes, if not none, might need to
            (nodes, edges, meta_nodes, start) = create_opo(
                nodes.clone(),
                edges.clone(),
                meta_nodes.clone(),
                &gromet.clone(),
                start.clone(),
                n1.clone(),
                bf_counter.clone(),
            );
            (nodes, edges, meta_nodes, start) = create_opi(
                nodes.clone(),
                edges.clone(),
                meta_nodes.clone(),
                &gromet.clone(),
                start.clone(),
                n1.clone(),
                bf_counter.clone(),
            );
            // now to construct the nodes inside the function, currently supported Literals and Primitives
            // first include an Expression for increased depth
            let mut box_counter: u8 = 1;
            for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
                match sboxf.function_type {
                    FunctionType::Function => {
                        (nodes, edges, start, meta_nodes) = create_function(
                            &gromet.clone(),                 // gromet for metadata
                            eboxf.value.clone(),             // function net for attribute
                            nodes.clone(),                   // nodes
                            edges.clone(),                   // edges
                            n1.clone(),                      // parent node
                            sboxf.contents.unwrap().clone(), // attribute idx
                            bf_counter.clone(),              // top level bf idx
                            box_counter.clone().into(),      // sub box idx, 0 for top level
                            start.clone(),                   // start
                            meta_nodes.clone(),              // metadata nodes
                        );
                    }

                    FunctionType::Predicate => {
                        (nodes, edges, start, meta_nodes) = create_att_predicate(
                            &gromet.clone(),
                            eboxf.clone(),
                            sboxf.clone(),
                            nodes.clone(),
                            edges.clone(),
                            n1.clone(),
                            att_idx.clone(),
                            box_counter.clone(),
                            bf_counter.clone(),
                            start.clone(),
                            meta_nodes.clone(),
                        );
                    }
                    FunctionType::Expression => {
                        (nodes, edges, start, meta_nodes) = create_att_expression(
                            &gromet.clone(),
                            eboxf.clone(),
                            sboxf.clone(),
                            nodes.clone(),
                            edges.clone(),
                            n1.clone(),
                            att_idx.clone(),
                            box_counter.clone(),
                            bf_counter.clone(),
                            start.clone(),
                            meta_nodes.clone(),
                        );
                    }
                    FunctionType::Literal => {
                        (nodes, edges, meta_nodes) = create_att_literal(
                            &gromet.clone(),
                            eboxf.clone(),
                            sboxf.clone(),
                            nodes.clone(),
                            edges.clone(),
                            n1.clone(),
                            att_idx.clone(),
                            box_counter.clone(),
                            bf_counter.clone(),
                            start.clone(),
                            meta_nodes.clone(),
                        );
                    }
                    FunctionType::Primitive => {
                        (nodes, edges, meta_nodes) = create_att_primitive(
                            &gromet.clone(),
                            eboxf.clone(),
                            sboxf.clone(),
                            nodes.clone(),
                            edges.clone(),
                            n1.clone(),
                            att_idx.clone(),
                            box_counter.clone(),
                            bf_counter.clone(),
                            start.clone(),
                            meta_nodes.clone(),
                        );
                    }
                    // stub for where to put in create function for nested functions
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
                att_idx.clone(),
                bf_counter.clone(),
            );
            // perform cross attributal wiring of function
            (edges, nodes) = cross_att_wiring(
                eboxf.clone(),
                nodes.clone(),
                edges,
                att_idx.clone(),
                bf_counter.clone(),
            );

            // make conditionals if they exist
            // might need to make wiring more robust
            if !gromet.modules[0].r#fn.bc.as_ref().is_none() {
                let mut cond_counter = 0;
                for _cond in gromet.modules[0].r#fn.bc.as_ref().unwrap().iter() {
                    // now lets check for and setup any conditionals at this level
                    (nodes, edges, start, meta_nodes) = create_conditional(
                        &gromet.clone(),
                        eboxf.value.clone(), // This is gromet but is more generalizable based on scope
                        nodes.clone(),
                        edges.clone(),
                        n1.clone(),
                        att_idx.clone() - 1, // because top level
                        cond_counter, // This indexes the conditional in the list of conditionals (bc)
                        bf_counter,   // because top level
                        start.clone(),
                        meta_nodes.clone(),
                    );
                    cond_counter += 1;
                }
            }
            // make loops if they exist
            // might need to make wiring more robust
            if !gromet.modules[0].r#fn.bl.as_ref().is_none() {
                let mut while_counter = 0;
                for _while_l in gromet.modules[0].r#fn.bl.as_ref().unwrap().iter() {
                    // now lets check for and setup any conditionals at this level
                    (nodes, edges, start, meta_nodes) = create_while_loop(
                        &gromet.clone(),
                        eboxf.value.clone(), // This is gromet but is more generalizable based on scope
                        nodes.clone(),
                        edges.clone(),
                        n1.clone(),
                        att_idx.clone() - 1, // because top level
                        while_counter, // This indexes the conditional in the list of conditionals (bc)
                        bf_counter,    // because top level
                        start.clone(),
                        meta_nodes.clone(),
                    );
                    while_counter += 1;
                }
            }
            return (nodes, edges, start, meta_nodes);
        }
        _ => return (nodes, edges, start, meta_nodes),
    }
}

// this creates the framework for conditionals, including the conditional node, the pic and poc nodes and the cond, body_if and body_else edges
// The iterator through the conditionals will need to be outside this funtion
#[allow(unused_assignments)]
pub fn create_conditional(
    gromet: &ModuleCollection,
    function_net: FunctionNet, // This is gromet but is more generalizable based on scope
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx_in: u32,       // This will index the attribute the conditional is in, if any
    cond_counter: u32, // This indexes the conditional in the list of conditionals (bc)
    bf_counter: u8,    // This indexes which box the conditional is under, if any
    mut start: u32,
    mut meta_nodes: Vec<MetadataNode>,
) -> (Vec<Node>, Vec<Edge>, u32, Vec<MetadataNode>) {
    let mut metadata_idx = 0;
    let n1 = Node {
        n_type: String::from("Conditional"),
        value: None,
        name: Some(format!("Conditional{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: idx_in + 1,
        nbox: bf_counter,
        att_bf_idx: 0,
    };
    let e1 = Edge {
        src: parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(cond_counter),
    };
    nodes.push(n1.clone());
    edges.push(e1);
    if !function_net.bc.as_ref().unwrap()[cond_counter as usize]
        .metadata
        .as_ref()
        .is_none()
    {
        metadata_idx = function_net.bc.as_ref().unwrap()[cond_counter as usize]
            .metadata
            .unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(
                &gromet.clone(),
                metadata_idx.clone(),
            ));
            let me1 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }

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
                att_bf_idx: 0,
            };
            let e3 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n2.clone());
            edges.push(e3);
            if !pic.metadata.as_ref().is_none() {
                metadata_idx = pic.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }

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
                att_bf_idx: 0,
            };
            let e5 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n3.clone());
            edges.push(e5);
            if !poc.metadata.as_ref().is_none() {
                metadata_idx = poc.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n3.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
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
    // now to perform the wl_cargs wiring which is a connection from the condition's pif/opi to the pic
    for wire in function_net.wl_cargs.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index, this points into bc I think
        let src_pif = function_net.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // attribute index of submodule (also opi contents value)
        let src_opi_idx = src_pif.id.unwrap().clone();
        let src_box = src_pif.r#box.clone(); // nbox value of src opi
        let src_att = function_net.bf.as_ref().unwrap()[(src_box - 1) as usize]
            .contents
            .unwrap()
            .clone();
        let src_nbox = src_box;

        let tgt_idx = wire.tgt; // port index
        let _tgt_pic = function_net.pic.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
        let tgt_box = bf_counter; // tgt sub module box number

        let tgt_att = idx_in + 1; // attribute index of submodule (also opo contents value)
        let tgt_nbox = tgt_box; // nbox value of tgt opo
        let tgt_pic_idx = tgt_idx;

        // now to construct the wire
        let mut wl_cargs_src_tgt: Vec<String> = vec![];
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
                                wl_cargs_src_tgt.push(node.node_id.clone());
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
                    if node.n_type == "Pic" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_pic_idx as u32) == *p {
                                wl_cargs_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if wl_cargs_src_tgt.len() == 2 {
            let e9 = Edge {
                src: wl_cargs_src_tgt[0].clone(),
                tgt: wl_cargs_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e9);
        }
    }

    // now to make the implicit wires that go from pics -> pifs/opis and pofs/opos -> pocs.
    // first we will iterate through the pics
    let mut pic_counter = 1;
    for _pic in function_net.pic.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        // Each pic is the target there will then be 2 srcs one for each wire, one going to "if" and one to "else"

        // first up we setup the if wire
        let src_if_box = function_net.bc.as_ref().unwrap()[0]
            .body_if
            .unwrap()
            .clone(); // get the box in the body if statement
        let src_if_att = function_net.bf.as_ref().unwrap()[(src_if_box - 1) as usize]
            .contents
            .unwrap()
            .clone(); // get the attribute this box lives in
        let src_if_nbox = src_if_box;
        let src_if_pif = gromet.modules[0].attributes[(src_if_att - 1) as usize]
            .value
            .opi
            .as_ref()
            .unwrap()[(pic_counter - 1) as usize]
            .clone(); // grab the pif that matches the pic were are on
        let src_if_opi_idx = src_if_pif.id.unwrap().clone();

        // now we setup the else wire
        let src_else_box = function_net.bc.as_ref().unwrap()[0]
            .body_else
            .unwrap()
            .clone(); // get the box in the body if statement
        let src_else_att = function_net.bf.as_ref().unwrap()[(src_else_box - 1) as usize]
            .contents
            .unwrap()
            .clone(); // get the attribute this box lives in
        let src_else_nbox = src_else_box;
        let src_else_pif = function_net.pif.as_ref().unwrap()[(pic_counter - 1) as usize].clone(); // grab the pif that matches the pic were are on
        let src_else_opi_idx = src_else_pif.id.unwrap().clone();

        // setting up the pic is straight forward
        let tgt_idx = pic_counter; // port index
        let tgt_box = bf_counter; // tgt sub module box number
        let tgt_att = idx_in + 1; // attribute index of submodule (also opo contents value)
        let tgt_nbox = tgt_box; // nbox value of tgt opo
        let tgt_pic_idx = tgt_idx;

        // now to construct the if wire
        let mut if_pic_src_tgt: Vec<String> = vec![];
        let mut else_pic_src_tgt: Vec<String> = vec![];
        // find the src if node
        for node in nodes.iter() {
            // make sure in correct box
            if src_if_nbox == (node.nbox as u32) {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_if_att == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_if_opi_idx as u32) == *p {
                                if_pic_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        // find the source else node
        for node in nodes.iter() {
            // make sure in correct box
            if src_else_nbox == (node.nbox as u32) {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_else_att == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_else_opi_idx as u32) == *p {
                                else_pic_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        // find the tgt pic that is the same for both wires
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att == node.contents {
                    // only opo's
                    if node.n_type == "Pic" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_pic_idx as u32) == *p {
                                if_pic_src_tgt.push(node.node_id.clone());
                                else_pic_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if if_pic_src_tgt.len() == 2 {
            let e10 = Edge {
                src: if_pic_src_tgt[0].clone(),
                tgt: if_pic_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e10);
        }
        if else_pic_src_tgt.len() == 2 {
            let e11 = Edge {
                src: else_pic_src_tgt[0].clone(),
                tgt: else_pic_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e11);
        }
        pic_counter += 1;
    }

    // now we construct the output wires for the bodies
    let mut poc_counter = 1;
    for _poc in function_net.poc.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        // Each pic is the target there will then be 2 srcs one for each wire, one going to "if" and one to "else"

        // first up we setup the if wire
        let tgt_if_box = function_net.bc.as_ref().unwrap()[0]
            .body_if
            .unwrap()
            .clone(); // get the box in the body if statement
        let tgt_if_att = function_net.bf.as_ref().unwrap()[(tgt_if_box - 1) as usize]
            .contents
            .unwrap()
            .clone(); // get the attribute this box lives in
        let tgt_if_nbox = tgt_if_box;
        let tgt_if_pof = gromet.modules[0].attributes[(tgt_if_att - 1) as usize]
            .value
            .opo
            .as_ref()
            .unwrap()[(poc_counter - 1) as usize]
            .clone(); // grab the pif that matches the pic were are on
        let tgt_if_opo_idx = tgt_if_pof.id.unwrap().clone();

        // now we setup the else wire
        let tgt_else_box = function_net.bc.as_ref().unwrap()[0]
            .body_else
            .unwrap()
            .clone(); // get the box in the body if statement
        let tgt_else_att = function_net.bf.as_ref().unwrap()[(tgt_else_box - 1) as usize]
            .contents
            .unwrap()
            .clone(); // get the attribute this box lives in
        let tgt_else_nbox = tgt_else_box;
        let tgt_else_pof = function_net.pof.as_ref().unwrap()[(pic_counter - 1) as usize].clone(); // grab the pif that matches the pic were are on
        let tgt_else_opo_idx = tgt_else_pof.id.unwrap().clone();

        // setting up the pic is straight forward
        let src_idx = poc_counter; // port index
        let src_box = bf_counter; // tgt sub module box number
        let src_att = idx_in + 1; // attribute index of submodule (also opo contents value)
        let src_nbox = src_box; // nbox value of tgt opo
        let src_poc_idx = src_idx;

        // now to construct the if wire
        let mut if_poc_src_tgt: Vec<String> = vec![];
        let mut else_poc_src_tgt: Vec<String> = vec![];
        // find the src if node
        for node in nodes.iter() {
            // make sure in correct box
            if src_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_att == node.contents {
                    // only opo's
                    if node.n_type == "Poc" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_poc_idx as u32) == *p {
                                if_poc_src_tgt.push(node.node_id.clone());
                                else_poc_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        // find the tgt else node
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_else_nbox == (node.nbox as u32) {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_else_att == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_else_opo_idx as u32) == *p {
                                else_poc_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        // find the source else node
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_if_nbox == (node.nbox as u32) {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_if_att == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_if_opo_idx as u32) == *p {
                                if_poc_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if if_poc_src_tgt.len() == 2 {
            let e12 = Edge {
                src: if_poc_src_tgt[0].clone(),
                tgt: if_poc_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e12);
        }
        if else_poc_src_tgt.len() == 2 {
            let e13 = Edge {
                src: else_poc_src_tgt[0].clone(),
                tgt: else_poc_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e13);
        }
        poc_counter += 1;
    }
    // might still need pass through wiring??

    return (nodes, edges, start, meta_nodes);
}
#[allow(unused_assignments)]
pub fn create_while_loop(
    gromet: &ModuleCollection,
    function_net: FunctionNet, // This is gromet but is more generalizable based on scope
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx_in: u32,       // This will index the attribute the conditional is in, if any
    cond_counter: u32, // This indexes the conditional in the list of conditionals (bc)
    bf_counter: u8,    // This indexes which box the conditional is under, if any
    mut start: u32,
    mut meta_nodes: Vec<MetadataNode>,
) -> (Vec<Node>, Vec<Edge>, u32, Vec<MetadataNode>) {
    let mut metadata_idx = 0;
    let n1 = Node {
        n_type: String::from("While_Loop"),
        value: None,
        name: Some(format!("While{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: idx_in + 1,
        nbox: bf_counter,
        att_bf_idx: 0,
    };
    let e1 = Edge {
        src: parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(cond_counter),
    };
    nodes.push(n1.clone());
    edges.push(e1);
    if !function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .metadata
        .as_ref()
        .is_none()
    {
        metadata_idx = function_net.bl.as_ref().unwrap()[cond_counter as usize]
            .metadata
            .unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(
                &gromet.clone(),
                metadata_idx.clone(),
            ));
            let me1 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }

    start += 1;

    // now we make the pic and poc ports and connect them to the conditional node
    if !function_net.pil.is_none() {
        let mut port_count = 1;
        for pic in function_net.pil.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut pic_name = String::from("Pil");
            if !pic.name.as_ref().is_none() {
                pic_name = pic.name.clone().unwrap();
            }
            // make the node
            // get the input ports
            let n2 = Node {
                n_type: String::from("Pil"),
                value: None,
                name: Some(pic_name),
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([port_count].to_vec()),
                contents: idx_in + 1,
                nbox: bf_counter,
                att_bf_idx: 0,
            };
            let e3 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n2.clone());
            edges.push(e3);
            if !pic.metadata.as_ref().is_none() {
                metadata_idx = pic.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            port_count += 1;
            start += 1;
        }
    }
    if !function_net.pol.is_none() {
        let mut port_count = 1;
        for poc in function_net.pol.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut poc_name = String::from("Pol");
            if !poc.name.as_ref().is_none() {
                poc_name = poc.name.clone().unwrap();
            }
            // make the node
            let n3 = Node {
                n_type: String::from("Pol"),
                value: None,
                name: Some(poc_name),
                node_id: format!("n{}", start),
                out_idx: Some([port_count].to_vec()),
                in_indx: None,
                contents: idx_in + 1,
                nbox: bf_counter,
                att_bf_idx: 0,
            };
            let e5 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n3.clone());
            edges.push(e5);
            if !poc.metadata.as_ref().is_none() {
                metadata_idx = poc.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n3.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            port_count += 1;
            start += 1;
        }
    }

    // Now let's create the edges for the conditional condition and it's bodies
    // find the nodes
    // the contents is the node's attribute reference, so need to pull off of box's contents value
    let mut condition_id = String::from("temp");
    let mut body_if_id = String::from("temp");
    let cond_box = function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .condition
        .clone()
        .unwrap();
    let body_if_box = function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .body
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
        e_type: String::from("Body"),
        prop: None,
    };
    edges.push(e8);

    // Now we start to wire these objects together there are two unique wire types and implicit wires that need to be made
    for wire in function_net.wfl.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_att = idx_in + 1; // attribute index of submodule (also opi contents value)
        let src_nbox = bf_counter; // nbox value of src opi
        let src_pil_idx = src_idx;

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
        let mut wfl_src_tgt: Vec<String> = vec![];
        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if src_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_att == node.contents {
                    // only opo's
                    if node.n_type == "Pil" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_pil_idx as u32) == *p {
                                wfl_src_tgt.push(node.node_id.clone());
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
                                wfl_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if wfl_src_tgt.len() == 2 {
            let e8 = Edge {
                src: wfl_src_tgt[0].clone(),
                tgt: wfl_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e8);
        }
    }
    // now to perform the wl_cargs wiring which is a connection from the condition's pif/opi to the pic
    for wire in function_net.wl_cargs.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index, this points into bc I think
        let src_pif = function_net.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // attribute index of submodule (also opi contents value)
        let src_opi_idx = src_pif.id.unwrap().clone();
        let src_box = src_pif.r#box.clone(); // nbox value of src opi
        let src_att = function_net.bf.as_ref().unwrap()[(src_box - 1) as usize]
            .contents
            .unwrap()
            .clone();
        let src_nbox = src_box;

        let tgt_idx = wire.tgt; // port index
        let _tgt_pil = function_net.pil.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
        let tgt_box = bf_counter; // tgt sub module box number

        let tgt_att = idx_in + 1; // attribute index of submodule (also opo contents value)
        let tgt_nbox = tgt_box; // nbox value of tgt opo
        let tgt_pil_idx = tgt_idx;

        // now to construct the wire
        let mut wl_cargs_src_tgt: Vec<String> = vec![];
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
                                wl_cargs_src_tgt.push(node.node_id.clone());
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
                    if node.n_type == "Pil" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_pil_idx as u32) == *p {
                                wl_cargs_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if wl_cargs_src_tgt.len() == 2 {
            let e9 = Edge {
                src: wl_cargs_src_tgt[0].clone(),
                tgt: wl_cargs_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e9);
        }
    }

    // now to make the implicit wires that go from pics -> pifs/opis and pofs/opos -> pocs.
    // first we will iterate through the pics
    let mut pic_counter = 1;
    for _pic in function_net.pil.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        // Each pic is the target there will then be 2 srcs one for each wire, one going to "if" and one to "else"

        // first up we setup the if wire
        let src_if_box = function_net.bl.as_ref().unwrap()[0].body.unwrap().clone(); // get the box in the body if statement
        let src_if_att = function_net.bf.as_ref().unwrap()[(src_if_box - 1) as usize]
            .contents
            .unwrap()
            .clone(); // get the attribute this box lives in
        let src_if_nbox = src_if_box;
        let src_if_pif = gromet.modules[0].attributes[(src_if_att - 1) as usize]
            .value
            .opi
            .as_ref()
            .unwrap()[(pic_counter - 1) as usize]
            .clone(); // grab the pif that matches the pic were are on
        let src_if_opi_idx = src_if_pif.id.unwrap().clone();

        // setting up the pic is straight forward
        let tgt_idx = pic_counter; // port index
        let tgt_box = bf_counter; // tgt sub module box number
        let tgt_att = idx_in + 1; // attribute index of submodule (also opo contents value)
        let tgt_nbox = tgt_box; // nbox value of tgt opo
        let tgt_pil_idx = tgt_idx;

        // now to construct the if wire
        let mut if_pil_src_tgt: Vec<String> = vec![];
        // find the src if node
        for node in nodes.iter() {
            // make sure in correct box
            if src_if_nbox == (node.nbox as u32) {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_if_att == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_if_opi_idx as u32) == *p {
                                if_pil_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        // find the tgt pic that is the same for both wires
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att == node.contents {
                    // only opo's
                    if node.n_type == "Pil" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_pil_idx as u32) == *p {
                                if_pil_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if if_pil_src_tgt.len() == 2 {
            let e10 = Edge {
                src: if_pil_src_tgt[0].clone(),
                tgt: if_pil_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e10);
        }
        pic_counter += 1;
    }

    // now we construct the output wires for the bodies
    let mut poc_counter = 1;
    for _poc in function_net.pol.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        // Each pic is the target there will then be 2 srcs one for each wire, one going to "if" and one to "else"

        // first up we setup the if wire
        let tgt_if_box = function_net.bl.as_ref().unwrap()[0].body.unwrap().clone(); // get the box in the body if statement
        let tgt_if_att = function_net.bf.as_ref().unwrap()[(tgt_if_box - 1) as usize]
            .contents
            .unwrap()
            .clone(); // get the attribute this box lives in
        let tgt_if_nbox = tgt_if_box;
        let tgt_if_pof = gromet.modules[0].attributes[(tgt_if_att - 1) as usize]
            .value
            .opo
            .as_ref()
            .unwrap()[(poc_counter - 1) as usize]
            .clone(); // grab the pif that matches the pic were are on
        let tgt_if_opo_idx = tgt_if_pof.id.unwrap().clone();

        // setting up the pic is straight forward
        let src_idx = poc_counter; // port index
        let src_box = bf_counter; // tgt sub module box number
        let src_att = idx_in + 1; // attribute index of submodule (also opo contents value)
        let src_nbox = src_box; // nbox value of tgt opo
        let src_pol_idx = src_idx;

        // now to construct the if wire
        let mut if_pol_src_tgt: Vec<String> = vec![];
        // find the src if node
        for node in nodes.iter() {
            // make sure in correct box
            if src_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_att == node.contents {
                    // only opo's
                    if node.n_type == "Pol" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_pol_idx as u32) == *p {
                                if_pol_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        // find the tgt node
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_if_nbox == (node.nbox as u32) {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_if_att == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_if_opo_idx as u32) == *p {
                                if_pol_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if if_pol_src_tgt.len() == 2 {
            let e12 = Edge {
                src: if_pol_src_tgt[0].clone(),
                tgt: if_pol_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e12);
        }
        poc_counter += 1;
    }
    // might still need pass through wiring??

    return (nodes, edges, start, meta_nodes);
}

// This needs to be updated to handle the new node structure and remove the overloaded contents field which will mess with the wiring alot
#[allow(unused_assignments)]
pub fn create_att_expression(
    gromet: &ModuleCollection,
    _eeboxf: Attribute,
    ssboxf: GrometBox,
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    _idx_in: u32,
    box_counter: u8,
    bf_counter: u8,
    mut start: u32,
    mut meta_nodes: Vec<MetadataNode>,
) -> (Vec<Node>, Vec<Edge>, u32, Vec<MetadataNode>) {
    let mut metadata_idx = 0;

    let idx = ssboxf.contents.unwrap(); // this is the attribute index of the current expression

    let n1 = Node {
        n_type: String::from("Expression"),
        value: None,
        name: Some(format!("Expression{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: idx, // this is the attribute index, USED TO BE PARENT ATTRIBUTE INDEX
        nbox: bf_counter, // inherited from parent
        att_bf_idx: box_counter as u32, // This is a reference to the index in parent function
    };
    let e1 = Edge {
        src: parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(idx),
    };
    nodes.push(n1.clone());
    edges.push(e1);
    if !ssboxf.metadata.as_ref().is_none() {
        metadata_idx = ssboxf.metadata.unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(
                &gromet.clone(),
                metadata_idx.clone(),
            ));
            let me1 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }
    // now travel to contents index of the attribute list (note it is 1 index,
    // so contents=1 => attribute[0])
    // create nodes and edges for this entry, include opo's and opi's
    start += 1;

    let eboxf = gromet.modules[0].attributes[(idx - 1) as usize].clone(); // current expression attribute
    let pboxf = gromet.modules[0].attributes[(parent_node.contents - 1) as usize].clone(); // parent attribute

    // construct opo nodes, if not none
    // not calling the opo port constuctors since they are based on grabbing in the name from top level of the gromet,
    // not a parental attribute
    if !eboxf.value.opo.clone().is_none() {
        // grab name which is one level up and based on indexing
        // this can be done by the parent nodes contents field should give the index of the attributes
        // this constructs a vec for the names in the pofs, if any.
        let mut opo_name: Vec<String> = vec![];
        for port in pboxf.value.pof.as_ref().unwrap().iter() {
            if port.r#box == box_counter {
                if !port.name.is_none() {
                    opo_name.push(port.name.as_ref().unwrap().clone());
                } else {
                    opo_name.push(String::from("un-named"));
                }
            }
        }
        if opo_name.clone().len() != 0 {
            let mut oport: u32 = 0;
            for _op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
                let n2 = Node {
                    n_type: String::from("Opo"),
                    value: None,
                    name: Some(opo_name[oport as usize].clone()),
                    node_id: format!("n{}", start),
                    out_idx: Some([oport + 1].to_vec()),
                    in_indx: None,
                    contents: idx,                  // current att idx
                    nbox: bf_counter,               // top parent top level idx
                    att_bf_idx: box_counter as u32, // current box idx of parent
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
                if !eboxf.value.opo.clone().as_ref().unwrap()[oport as usize]
                    .metadata
                    .clone()
                    .as_ref()
                    .is_none()
                {
                    metadata_idx = eboxf.value.opo.clone().unwrap()[oport as usize]
                        .metadata
                        .clone()
                        .unwrap();
                    let mut repeat_meta = false;
                    for node in meta_nodes.iter() {
                        if node.metadata_idx == metadata_idx {
                            repeat_meta = true;
                        }
                    }
                    if !repeat_meta {
                        meta_nodes.append(&mut create_metadata_node(
                            &gromet.clone(),
                            metadata_idx.clone(),
                        ));
                        let me1 = Edge {
                            src: n2.node_id.clone(),
                            tgt: format!("m{}", metadata_idx),
                            e_type: String::from("Metadata"),
                            prop: None,
                        };
                        edges.push(me1);
                    }
                }
                // construct any metadata edges
                start += 1;
                oport += 1;
            }
        }
    }
    // construct opi nodes, in not none
    if !eboxf.value.opi.clone().is_none() {
        // grab name which is NOT one level up as in opo
        let mut opi_name: Vec<String> = vec![];
        for port in eboxf.value.opi.as_ref().unwrap().iter() {
            if !port.name.is_none() {
                opi_name.push(port.name.as_ref().unwrap().clone());
            } else {
                opi_name.push(String::from("un-named"));
            }
        }
        let mut iport: u32 = 0;
        for _op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opi"),
                value: None,
                name: Some(opi_name[iport as usize].clone()), // I think this naming will get messed up if there are multiple ports...
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([iport + 1].to_vec()),
                contents: idx,
                nbox: bf_counter,
                att_bf_idx: box_counter as u32,
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
            if !eboxf.value.opi.clone().as_ref().unwrap()[iport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_none()
            {
                metadata_idx = eboxf.value.opi.clone().unwrap()[iport as usize]
                    .metadata
                    .clone()
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            start += 1;
            iport += 1;
        }
    }
    // now to construct the nodes inside the expression, Literal and Primitives
    let mut box_counter: u8 = 1;
    if !eboxf.value.bf.is_none() {
        for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
            match sboxf.function_type {
                FunctionType::Literal => {
                    (nodes, edges, meta_nodes) = create_att_literal(
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
                        meta_nodes.clone(),
                    );
                }
                FunctionType::Primitive => {
                    (nodes, edges, meta_nodes) = create_att_primitive(
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
                        meta_nodes.clone(),
                    );
                }
                _ => {}
            }
            box_counter += 1;
            start += 1;
        }
    }
    // Now we perform the internal wiring of this branch
    edges = internal_wiring(
        eboxf.clone(),
        nodes.clone(),
        edges,
        idx.clone(),
        bf_counter.clone(),
    );

    // Now we also perform wopio wiring in case there is an empty expression
    if !eboxf.value.wopio.is_none() {
        edges = wopio_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone() - 1,
            bf_counter.clone(),
        );
    }

    return (nodes, edges, start, meta_nodes);
}

// This needs to be updated to handle the new node structure and remove the overloaded contents field which will mess with the wiring alot
#[allow(unused_assignments)]
pub fn create_att_predicate(
    gromet: &ModuleCollection,
    _eeboxf: Attribute,
    ssboxf: GrometBox,
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    _idx_in: u32,
    box_counter: u8,
    bf_counter: u8,
    mut start: u32,
    mut meta_nodes: Vec<MetadataNode>,
) -> (Vec<Node>, Vec<Edge>, u32, Vec<MetadataNode>) {
    let mut metadata_idx = 0;

    let idx = ssboxf.contents.unwrap(); // this is the attribute index of the current expression

    let n1 = Node {
        n_type: String::from("Predicate"),
        value: None,
        name: Some(format!("Predicate{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: idx, // this is the attribute index, USED TO BE PARENT ATTRIBUTE INDEX
        nbox: bf_counter, // inherited from parent
        att_bf_idx: box_counter as u32, // This is a reference to the index in parent function
    };
    let e1 = Edge {
        src: parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(idx),
    };
    nodes.push(n1.clone());
    edges.push(e1);
    if !ssboxf.metadata.as_ref().is_none() {
        metadata_idx = ssboxf.metadata.unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(
                &gromet.clone(),
                metadata_idx.clone(),
            ));
            let me1 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }
    // now travel to contents index of the attribute list (note it is 1 index,
    // so contents=1 => attribute[0])
    // create nodes and edges for this entry, include opo's and opi's
    start += 1;

    let eboxf = gromet.modules[0].attributes[(idx - 1) as usize].clone(); // current expression attribute
    let pboxf = gromet.modules[0].attributes[(parent_node.contents - 1) as usize].clone(); // parent attribute

    // construct opo nodes, if not none
    // not calling the opo port constuctors since they are based on grabbing in the name from top level of the gromet,
    // not a parental attribute
    if !eboxf.value.opo.clone().is_none() {
        // grab name which is one level up and based on indexing
        // this can be done by the parent nodes contents field should give the index of the attributes
        // this constructs a vec for the names in the pofs, if any.
        let mut opo_name: Vec<String> = vec![];
        for port in pboxf.value.pof.as_ref().unwrap().iter() {
            if port.r#box == box_counter {
                if !port.name.is_none() {
                    opo_name.push(port.name.as_ref().unwrap().clone());
                } else {
                    opo_name.push(String::from("un-named"));
                }
            }
        }
        let mut oport: u32 = 0;
        for _op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opo"),
                value: None,
                name: Some(opo_name[oport as usize].clone()),
                node_id: format!("n{}", start),
                out_idx: Some([oport + 1].to_vec()),
                in_indx: None,
                contents: idx,                  // current att idx
                nbox: bf_counter,               // top parent top level idx
                att_bf_idx: box_counter as u32, // current box idx of parent
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
            if !eboxf.value.opo.clone().as_ref().unwrap()[oport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_none()
            {
                metadata_idx = eboxf.value.opo.clone().unwrap()[oport as usize]
                    .metadata
                    .clone()
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            // construct any metadata edges
            start += 1;
            oport += 1;
        }
    }
    // construct opi nodes, in not none
    if !eboxf.value.opi.clone().is_none() {
        // grab name which is NOT one level up as in opo
        let mut opi_name: Vec<String> = vec![];
        for port in eboxf.value.opi.as_ref().unwrap().iter() {
            if !port.name.is_none() {
                opi_name.push(port.name.as_ref().unwrap().clone());
            } else {
                opi_name.push(String::from("un-named"));
            }
        }
        let mut iport: u32 = 0;
        for _op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opi"),
                value: None,
                name: Some(opi_name[iport as usize].clone()), // I think this naming will get messed up if there are multiple ports...
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([iport + 1].to_vec()),
                contents: idx,
                nbox: bf_counter,
                att_bf_idx: box_counter as u32,
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
            if !eboxf.value.opi.clone().as_ref().unwrap()[iport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_none()
            {
                metadata_idx = eboxf.value.opi.clone().unwrap()[iport as usize]
                    .metadata
                    .clone()
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            start += 1;
            iport += 1;
        }
    }
    // now to construct the nodes inside the expression, Literal and Primitives
    let mut box_counter: u8 = 1;
    for sboxf in eboxf.value.bf.clone().as_ref().unwrap().iter() {
        match sboxf.function_type {
            FunctionType::Literal => {
                (nodes, edges, meta_nodes) = create_att_literal(
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
                    meta_nodes.clone(),
                );
            }
            FunctionType::Primitive => {
                (nodes, edges, meta_nodes) = create_att_primitive(
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
                    meta_nodes.clone(),
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
    return (nodes, edges, start, meta_nodes);
}
#[allow(unused_assignments)]
pub fn create_att_literal(
    gromet: &ModuleCollection,
    eboxf: Attribute,
    sboxf: GrometBox,
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx: u32,
    box_counter: u8,
    bf_counter: u8,
    start: u32,
    mut meta_nodes: Vec<MetadataNode>,
) -> (Vec<Node>, Vec<Edge>, Vec<MetadataNode>) {
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
    let mut metadata_idx = 0;
    let n3 = Node {
        n_type: String::from("Literal"),
        value: Some(format!("{:?}", sboxf.value.clone().as_ref().unwrap())),
        name: None,
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: None, // literals should only have out ports
        contents: idx,
        nbox: bf_counter,
        att_bf_idx: box_counter as u32,
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
    if !sboxf.metadata.is_none() {
        metadata_idx = sboxf.metadata.clone().unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(
                &gromet.clone(),
                metadata_idx.clone(),
            ));
            let me1 = Edge {
                src: n3.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }
    return (nodes, edges, meta_nodes);
}
#[allow(unused_assignments)]
pub fn create_att_primitive(
    gromet: &ModuleCollection,
    eboxf: Attribute,
    sboxf: GrometBox,
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    parent_node: Node,
    idx: u32,
    box_counter: u8,
    bf_counter: u8,
    start: u32,
    mut meta_nodes: Vec<MetadataNode>,
) -> (Vec<Node>, Vec<Edge>, Vec<MetadataNode>) {
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
    }
    // now make the node with the port information
    let mut metadata_idx = 0;
    let n3 = Node {
        n_type: String::from("Primitive"),
        value: None,
        name: sboxf.name.clone(),
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: Some(pif),
        contents: idx,
        nbox: bf_counter,
        att_bf_idx: box_counter as u32,
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
    if !sboxf.metadata.is_none() {
        metadata_idx = sboxf.metadata.clone().unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(
                &gromet.clone(),
                metadata_idx.clone(),
            ));
            let me1 = Edge {
                src: n3.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }
    return (nodes, edges, meta_nodes);
}
// This is for the construction of Opo's
#[allow(unused_assignments)]
pub fn create_opo(
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    mut meta_nodes: Vec<MetadataNode>,
    gromet: &ModuleCollection,
    mut start: u32,
    parent_node: Node,
    box_counter: u8,
) -> (Vec<Node>, Vec<Edge>, Vec<MetadataNode>, u32) {
    let eboxf = gromet.modules[0].attributes[(parent_node.contents - 1) as usize].clone();
    // construct opo nodes, if not none
    if !eboxf.value.opo.clone().is_none() {
        // grab name which is one level up and based on indexing
        let mut opo_name = "un-named";
        let mut oport: u32 = 0;
        for op in eboxf.value.opo.clone().as_ref().unwrap().iter() {
            if op.name.as_ref().is_none() && !gromet.modules[0].r#fn.pof.as_ref().is_none() {
                for port in gromet.modules[0].r#fn.pof.as_ref().unwrap().iter() {
                    if port.r#box == box_counter {
                        if oport == (port.id.unwrap() as u32 - 1) {
                            if !port.name.is_none() {
                                opo_name = port.name.as_ref().unwrap();
                            }
                        }
                    }
                }
            } else if !op.name.as_ref().is_none() {
                opo_name = op.name.as_ref().unwrap();
            }
            let n2 = Node {
                n_type: String::from("Opo"),
                value: None,
                name: Some(String::from(opo_name)),
                node_id: format!("n{}", start),
                out_idx: Some([oport + 1].to_vec()),
                in_indx: None,
                contents: parent_node.contents.clone(),
                nbox: box_counter.clone(),
                att_bf_idx: parent_node.att_bf_idx.clone(),
            };
            nodes.push(n2.clone());
            // construct edge: expression -> Opo
            let e3 = Edge {
                src: parent_node.node_id.clone(),
                tgt: n2.node_id.clone(),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            edges.push(e3);

            let mut metadata_idx = 0;
            if !eboxf.value.opo.clone().as_ref().unwrap()[oport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_none()
            {
                metadata_idx = eboxf.value.opo.clone().unwrap()[oport as usize]
                    .metadata
                    .clone()
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            // construct any metadata edges
            start += 1;
            oport += 1;
        }
    }
    return (nodes, edges, meta_nodes, start);
}

// This is for the construction of Opi's
#[allow(unused_assignments)]
pub fn create_opi(
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    mut meta_nodes: Vec<MetadataNode>,
    gromet: &ModuleCollection,
    mut start: u32,
    parent_node: Node,
    box_counter: u8,
) -> (Vec<Node>, Vec<Edge>, Vec<MetadataNode>, u32) {
    let eboxf = gromet.modules[0].attributes[(parent_node.contents - 1) as usize].clone();
    // construct opi nodes, if not none
    if !eboxf.value.opi.clone().is_none() {
        // grab name which is one level up and based on indexing
        let mut opi_name = "un-named";
        let mut oport: u32 = 0;
        for op in eboxf.value.opi.clone().as_ref().unwrap().iter() {
            if op.name.as_ref().is_none() && !gromet.modules[0].r#fn.pif.as_ref().is_none() {
                for port in gromet.modules[0].r#fn.pif.as_ref().unwrap().iter() {
                    if port.r#box == box_counter {
                        if oport == (port.id.unwrap() as u32 - 1) {
                            if !port.name.is_none() {
                                opi_name = port.name.as_ref().unwrap();
                            }
                        }
                    }
                }
            } else if !op.name.as_ref().is_none() {
                opi_name = op.name.as_ref().unwrap();
            }
            let n2 = Node {
                n_type: String::from("Opi"),
                value: None,
                name: Some(String::from(opi_name)),
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([oport + 1].to_vec()),
                contents: parent_node.contents.clone(),
                nbox: box_counter.clone(),
                att_bf_idx: parent_node.att_bf_idx.clone(),
            };
            nodes.push(n2.clone());
            // construct edge: expression <- Opi
            let e3 = Edge {
                src: parent_node.node_id.clone(),
                tgt: n2.node_id.clone(),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            edges.push(e3);

            let mut metadata_idx = 0;
            if !eboxf.value.opi.clone().as_ref().unwrap()[oport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_none()
            {
                metadata_idx = eboxf.value.opi.clone().unwrap()[oport as usize]
                    .metadata
                    .clone()
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(
                        &gromet.clone(),
                        metadata_idx.clone(),
                    ));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            // construct any metadata edges
            start += 1;
            oport += 1;
        }
    }
    return (nodes, edges, meta_nodes, start);
}
// having issues with deeply nested structure, it is breaking in the internal wiring of the function level.
// wfopi: pif -> opi
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
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx) == node.contents {
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
                if (idx) == node.contents {
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
                if (idx) == node.contents {
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
                if (idx) == node.contents {
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
// this is duplicating wires a ton. (01/15/23)
// shouldn't use bf_counter, should use att_bf_idx since that is variable for functions, need to pull the box from the
// ports the wires point to.
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

        let src_idx = wire.src; // port index

        let src_pif = eboxf.value.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port

        let src_box = src_pif.r#box.clone(); // src sub module box number
        let src_att = idx; // attribute index of submodule (also opi contents value)
        let src_nbox = bf_counter; // nbox value of src opi

        let tgt_idx = wire.tgt; // port index

        let tgt_pof = eboxf.value.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port

        let tgt_box = tgt_pof.r#box.clone(); // tgt sub module box number
        let tgt_att = idx; // attribute index of submodule (also opo contents value)
        let tgt_nbox = bf_counter; // nbox value of tgt opo

        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if src_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_att == node.contents {
                    // matche the box
                    if (src_box as u32) == node.att_bf_idx {
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
        }
        // finding the tgt node
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_nbox == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att == node.contents {
                    // match internal box
                    if (tgt_box as u32) == node.att_bf_idx {
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

pub fn wopio_wiring(
    eboxf: Attribute,
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,
    bf_counter: u8,
) -> Vec<Edge> {
    // iterate through all wires of type
    for wire in eboxf.value.wopio.unwrap().iter() {
        let mut wopio_src_tgt: Vec<String> = vec![];
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
                                wopio_src_tgt.push(node.node_id.clone());
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
                    if node.out_idx.is_none() {
                        // only opi's
                        if node.n_type == "Opi" {
                            // iterate through port to check for src
                            for p in node.in_indx.as_ref().unwrap().iter() {
                                // push the tgt
                                if (wire.tgt as u32) == *p {
                                    wopio_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
        if wopio_src_tgt.len() == 2 {
            let e7 = Edge {
                src: wopio_src_tgt[0].clone(),
                tgt: wopio_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e7);
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
    // wopio: opo -> opi

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

    // check if wire exists, wopio
    if !eboxf.value.wopio.is_none() {
        edges = wopio_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone(),
            bf_counter.clone(),
        );
    }

    return edges;
}

// now for the wiring used for imports
// needs to handle top level and function level wiring that uses the function net at the call of the import.
pub fn import_wiring(
    gromet: &ModuleCollection,
    _eboxf: Attribute,
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,
    bf_counter: u8,
    parent_node: Node,
) -> Vec<Edge> {
    // first based on the parent_node determine if we need to grab the pof's from the top scope or a sub scope
    if parent_node.att_bf_idx == 0 {
        // this means top level wiring
        // iterate through all wires of type
        let nboxf = gromet.modules[0].r#fn.clone();
        for wire in nboxf.wff.unwrap().iter() {
            let mut wff_src_tgt: Vec<String> = vec![];

            let src_idx = wire.src; // port index

            let src_pif = nboxf.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port

            let src_box = src_pif.r#box.clone(); // src sub module box number
            let src_att = idx; // attribute index of submodule (also opi contents value)
            let src_nbox = bf_counter; // nbox value of src opi

            let tgt_idx = wire.tgt; // port index

            let tgt_pof = nboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port

            let tgt_box = tgt_pof.r#box.clone(); // tgt sub module box number
            let tgt_att = idx; // attribute index of submodule (also opo contents value)
            let tgt_nbox = bf_counter; // nbox value of tgt opo

            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if src_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if src_att == node.contents {
                        // matche the box
                        if (src_box as u32) == node.att_bf_idx {
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
            }
            // finding the tgt node
            for node in nodes.iter() {
                // make sure in correct box
                if tgt_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att == node.contents {
                        // match internal box
                        if (tgt_box as u32) == node.att_bf_idx {
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
    } else {
        // this means we are in function scope, concerns on if this is cross attributal or just internal wiring...
        let eboxf = gromet.modules[0].attributes[(parent_node.contents - 1) as usize].clone();

        // iterate through all wires of type
        for wire in eboxf.value.wff.unwrap().iter() {
            let mut wff_src_tgt: Vec<String> = vec![];

            let src_idx = wire.src; // port index

            let src_pif = eboxf.value.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port

            let src_box = src_pif.r#box.clone(); // src sub module box number
            let src_att = idx; // attribute index of submodule (also opi contents value)
            let src_nbox = bf_counter; // nbox value of src opi

            let tgt_idx = wire.tgt; // port index

            let tgt_pof = eboxf.value.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port

            let tgt_box = tgt_pof.r#box.clone(); // tgt sub module box number
            let tgt_att = idx; // attribute index of submodule (also opo contents value)
            let tgt_nbox = bf_counter; // nbox value of tgt opo

            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if src_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if src_att == node.contents {
                        // matche the box
                        if (src_box as u32) == node.att_bf_idx {
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
            }
            // finding the tgt node
            for node in nodes.iter() {
                // make sure in correct box
                if tgt_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att == node.contents {
                        // match internal box
                        if (tgt_box as u32) == node.att_bf_idx {
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
}

pub fn cross_att_wiring(
    eboxf: Attribute, // This is the current attribute
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,       // this +1 is the current attribute index
    bf_counter: u8, // this is the current box
) -> (Vec<Edge>, Vec<Node>) {
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
        (edges, nodes) = wff_cross_att_wiring(
            eboxf.clone(),
            nodes.clone(),
            edges,
            idx.clone(),
            bf_counter.clone(),
        );
    }
    return (edges, nodes);
}
// this will construct connections from the function opi's to the sub module opi's, tracing inputs through the function
// opi(sub)->opi(fun)
pub fn wfopi_cross_att_wiring(
    eboxf: Attribute, // This is the current attribute
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,       // this is the current attribute index
    bf_counter: u8, // this is the current box
) -> Vec<Edge> {
    for wire in eboxf.value.wfopi.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_pif = eboxf.value.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port
        let src_opi_idx = src_pif.id.unwrap().clone(); // index of opi port in opi list in src sub module (also opi node in_indx value)
        let src_box = src_pif.r#box.clone(); // src sub module box number, should also be the att_bf_idx of the opi

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
            let tgt_att = idx; // attribute index of function
            let tgt_nbox = bf_counter; // nbox value of tgt opi

            // now to construct the wire
            let mut wfopi_src_tgt: Vec<String> = vec![];
            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if src_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if src_att == node.contents {
                        // make sure box index also lines up
                        if (src_box as u32) == node.att_bf_idx {
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
            let src_att = idx; // attribute index of function
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
                        if (tgt_box as u32) == node.att_bf_idx {
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
#[allow(unused_assignments)]
pub fn wff_cross_att_wiring(
    eboxf: Attribute, // This is the current attribute, should be the function if in a function
    mut nodes: Vec<Node>,
    mut edges: Vec<Edge>,
    idx: u32,       // this +1 is the current attribute index
    bf_counter: u8, // this is the current box
) -> (Vec<Edge>, Vec<Node>) {
    for wire in eboxf.value.wff.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_pif = eboxf.value.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port
        let src_opi_idx = src_pif.id.unwrap().clone(); // index of opi port in opi list in src sub module (also opi node in_indx value)
        let src_box = src_pif.r#box.clone(); // src sub module box number
                                             // make sure its a cross attributal wiring and not an internal wire
        let mut src_att = idx;
        // first conditional for if we are wiring from expression or function
        if !eboxf.value.bf.as_ref().unwrap()[(src_box - 1) as usize]
            .contents
            .clone()
            .is_none()
        {
            src_att = eboxf.value.bf.as_ref().unwrap()[(src_box - 1) as usize]
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
            // initialize the tgt_att for case of opo or primitive/literal source
            let mut tgt_att = idx;
            // next expression for if we are wiring to an expression or function
            if !eboxf.value.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .contents
                .clone()
                .is_none()
            {
                tgt_att = eboxf.value.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
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
                            if (src_box as u32) == node.att_bf_idx {
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
                }
                for node in nodes.iter() {
                    // make sure in correct box
                    if tgt_nbox == node.nbox {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if tgt_att == node.contents {
                            if (tgt_box as u32) == node.att_bf_idx {
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
            else 
            {
                let tgt_nbox = bf_counter; // nbox value of tgt opo
                                           // now to construct the wire
                let mut wff_src_tgt: Vec<String> = vec![];
                // find the src node
                for node in nodes.iter() {
                    // make sure in correct box
                    if src_nbox == node.nbox {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if src_att == node.contents {
                            if (src_box as u32) == node.att_bf_idx {
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
                }
                for node in nodes.iter() {
                    // make sure in correct box
                    if tgt_nbox == node.nbox {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if tgt_att == node.contents {
                            if (tgt_box as u32) == node.att_bf_idx {
                                // only opo's
                                if node.n_type == "Primitive" || node.n_type == "Literal" {
                                    // iterate through port to check for tgt
                                    for p in node.out_idx.as_ref().unwrap().iter() {
                                        // push the src first, being pif
                                        if (tgt_idx as u32) == *p {
                                            wff_src_tgt.push(node.node_id.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                if wff_src_tgt.len() == 2 {
                    // now we perform a conditional for naming the opi's based on 
                    // the primitives pof names, we have the primitive, the opi node id
                    // and the tgt_opo_idx which is the pof idx for the name
                    for i in 0..nodes.clone().len() {
                        if nodes[i].node_id.clone() == wff_src_tgt[0].clone() {
                            if !eboxf.value.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].name.is_none() {
                                nodes[i].name = eboxf.value.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].name.clone();
                            }
                        }
                    }
                    let e8 = Edge {
                        src: wff_src_tgt[0].clone(),
                        tgt: wff_src_tgt[1].clone(),
                        e_type: String::from("Wire"),
                        prop: None,
                    };
                    edges.push(e8);
                }
            }
        } else {
                let src_nbox = bf_counter; // nbox value of src opi
                                           // collect info to identify the opo tgt node
                let tgt_idx = wire.tgt; // port index
                let tgt_pof = eboxf.value.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
                let tgt_opo_idx = tgt_pof.id.unwrap().clone(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
                let tgt_box = tgt_pof.r#box.clone(); // tgt sub module box number
                                                     // make sure its a cross attributal wiring and not an internal wire
                // initialize the tgt_att for case of opo or primitive/literal source
                let mut tgt_att = idx;
                if !eboxf.value.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                    .contents
                    .clone()
                    .is_none()
                {
                    tgt_att = eboxf.value.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
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
                                if (src_box as u32) == node.att_bf_idx {
                                    // only opo's
                                    if node.n_type == "Primitive" {
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
                    }
                    for node in nodes.iter() {
                        // make sure in correct box
                        if tgt_nbox == node.nbox {
                            // make sure only looking in current attribute nodes for srcs and tgts
                            if tgt_att == node.contents {
                                if (tgt_box as u32) == node.att_bf_idx {
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
    return (edges, nodes);
}
// external wiring is the wiring between boxes at the module level
pub fn external_wiring(
    gromet: &ModuleCollection,
    nodes: Vec<Node>,
    mut edges: Vec<Edge>,
) -> Vec<Edge> {
    if !gromet.modules[0].r#fn.wff.as_ref().is_none() {
        for wire in gromet.modules[0].r#fn.wff.as_ref().unwrap().iter() {
            let src_idx = wire.src; // pif wire connects to
            let tgt_idx = wire.tgt; // pof wire connects to
            let src_id = gromet.modules[0].r#fn.pif.as_ref().unwrap()[(src_idx - 1) as usize]
                .id
                .unwrap(); // pif id
            let src_box =
                gromet.modules[0].r#fn.pif.as_ref().unwrap()[(src_idx - 1) as usize].r#box; // pif box
            let mut src_att = 0;
            if gromet.modules[0].r#fn.bf.as_ref().unwrap()[(src_box - 1) as usize]
                .function_type
                .clone()
                == FunctionType::Function
                || gromet.modules[0].r#fn.bf.as_ref().unwrap()[(src_box - 1) as usize]
                    .function_type
                    .clone()
                    == FunctionType::Expression
            {
                src_att = gromet.modules[0].r#fn.bf.as_ref().unwrap()[(src_box - 1) as usize]
                    .contents
                    .unwrap()
                    .clone();
            }
            let tgt_id = gromet.modules[0].r#fn.pof.as_ref().unwrap()[(tgt_idx - 1) as usize]
                .id
                .unwrap(); // pof id
            let tgt_box =
                gromet.modules[0].r#fn.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].r#box; // pof box
            let mut tgt_att = 0;
            if gromet.modules[0].r#fn.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .function_type
                .clone()
                == FunctionType::Function
                || gromet.modules[0].r#fn.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                    .function_type
                    .clone()
                    == FunctionType::Expression
            {
                tgt_att = gromet.modules[0].r#fn.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                    .contents
                    .unwrap()
                    .clone();
            }
            let mut wff_src_tgt: Vec<String> = vec![];
            // This is double counting since only check is name and box, check on attributes?
            // find the src
            for node in nodes.iter() {
                if node.nbox == src_box {
                    if src_att == node.contents {
                        if node.n_type == "Opi" || node.n_type == "Import" {
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
            // find the tgt
            let mut tgt_found = false; // for expression wiring the literal and opo's can be double counted sometimes
            for node in nodes.iter() {
                // check this field
                if node.n_type == "Opo" {
                    if node.nbox == tgt_box {
                        if tgt_att == node.contents {
                            for p in node.out_idx.as_ref().unwrap().iter() {
                                // push the tgt
                                if (tgt_id as u32) == *p {
                                    wff_src_tgt.push(node.node_id.clone());
                                    tgt_found = true;
                                }
                            }
                        } else {
                            // perform extra check to make sure not getting sub cross attributal nodes
                            if tgt_att == node.contents {
                                for p in node.out_idx.as_ref().unwrap().iter() {
                                    // push the tgt
                                    if (tgt_id as u32) == *p {
                                        wff_src_tgt.push(node.node_id.clone());
                                        tgt_found = true;
                                    }
                                }
                            }
                        }
                    }
                } else if (node.n_type == "Literal" || node.n_type == "Import") && !tgt_found {
                    if node.nbox == tgt_box {
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the tgt
                            if (tgt_id as u32) == *p {
                                wff_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
            if wff_src_tgt.len() == 2 {
                let e9 = Edge {
                    src: wff_src_tgt[0].clone(),
                    tgt: wff_src_tgt[1].clone(),
                    e_type: String::from("Wire"),
                    prop: None,
                };
                edges.push(e9);
            }
        }
    }
    return edges;
}

pub fn parse_gromet_queries(gromet: ModuleCollection) -> Vec<String> {
    let mut queries: Vec<String> = vec![];

    let start: u32 = 0;

    queries.append(&mut create_module(&gromet));
    queries.append(&mut create_graph_queries(&gromet, start));

    return queries;
}
