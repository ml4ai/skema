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

/* 3/20/23
   - '+' at the top level main are getting duplicate wires to the top level literals
*/
use crate::FunctionType;
use crate::{Files, Grounding, ModuleCollection, Provenance, TextExtraction, ValueMeta};
use crate::{FunctionNet, GrometBox, ValueL};
use rsmgclient::{ConnectParams, Connection, MgError};
use crate::config::Config;

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
    pub value: Option<ValueL>,
    pub name: Option<String>,
    pub node_id: String,
    pub out_idx: Option<Vec<u32>>, // opo or pof index, directly matching the wire src/tgt notation
    pub in_indx: Option<Vec<u32>>, // opi or pif index
    pub contents: usize, // This indexes which index this node has inside the fn_array list, but for functions this is overloaded
    pub nbox: usize, // this indexs the top level box call for the node, if any, default for none is 0
    pub att_bf_idx: usize, // this indexes the attribute index of one higher scope if any, default for none is 0
    pub box_counter: usize, // this indexes the box call for the node one scope up, matches nbox if higher scope is top level
}

#[derive(Debug, Clone, PartialEq, Ord, Eq, PartialOrd)]
pub struct Edge {
    pub src: String,
    pub tgt: String,
    pub e_type: String,
    pub prop: Option<usize>, // option because of opo's and opi's
}

#[derive(Debug, Clone)]
pub struct ConstructorArgs {
    pub att_box: FunctionNet, // current attribute object is in
    pub cur_box: GrometBox,   // current box object is in
    pub parent_node: Node,    // parent node of the current constructor
    pub att_idx: usize,       // This will index the attribute the function is in, 0 if not
    pub bf_counter: usize, // This indexes which top level box the function is under, inherited from parent if not explicit, 0 if not
    pub att_bf_idx: usize, // This indexes if the function is a subscope of a larger function, 0 if not
    pub box_counter: usize, // this is the index of the box if called inside another function, 0 if not
}

pub fn execute_query(query: &str, config: Config) -> Result<(), MgError> {
    // Connect to Memgraph.
    let connect_params = ConnectParams {
        port: config.db_port,
        host: Some(format!("{}{}", config.db_proto.clone(), config.db_host.clone())),
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
        [(metadata_idx - 1) as usize][0]
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
    for data in
        gromet.modules[0].metadata_collection.as_ref().unwrap()[(metadata_idx - 1) as usize].clone()
    {
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
        line_begin_vec.push(data.line_begin);
        line_end_vec.push(data.line_end);
        col_begin_vec.push(data.col_begin);
        col_end_vec.push(data.col_end);
        provenance_vec.push(data.provenance.clone());
    }

    let m1 = MetadataNode {
        n_type: String::from("Metadata"),
        node_id: format!("m{}", metadata_idx),
        metadata_idx,
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
    metas
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
        metadata_type_q.push(
            meta_node.metadata_type[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        gromet_version_q.push(
            meta_node.gromet_version[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        text_extraction_q.push(
            meta_node.text_extraction[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| format!("{:?}", x)),
        );
        variable_identifier_q.push(
            meta_node.variable_identifier[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        variable_definition_q.push(
            meta_node.variable_definition[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        value_q.push(
            meta_node.value[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| format!("{:?}", x)),
        );
        grounding_q.push(
            meta_node.grounding[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| format!("{:?}", x)),
        );
        name_q.push(
            meta_node.name[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        global_reference_id_q.push(
            meta_node.global_reference_id[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        files_q.push(
            meta_node.files[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| format!("{:?}", x)),
        );
        source_language_q.push(
            meta_node.source_language[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        source_language_version_q.push(
            meta_node.source_language_version[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        data_type_q.push(
            meta_node.data_type[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        code_file_reference_uid_q.push(
            meta_node.code_file_reference_uid[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| x.to_string()),
        );
        line_begin_q.push(meta_node.line_begin[i].as_ref().map_or_else(|| 0, |x| *x));
        line_end_q.push(meta_node.line_end[i].as_ref().map_or_else(|| 0, |x| *x));
        col_begin_q.push(meta_node.col_begin[i].as_ref().map_or_else(|| 0, |x| *x));
        col_end_q.push(meta_node.col_end[i].as_ref().map_or_else(|| 0, |x| *x));
        provenance_q.push(
            meta_node.provenance[i]
                .as_ref()
                .map_or_else(|| String::from(""), |x| format!("{:?}", x)),
        );
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
    queries
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

    let meta = create_metadata_node(gromet, *metadata_idx);
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

    queries
}

fn create_graph_queries(gromet: &ModuleCollection, start: u32) -> Vec<String> {
    let mut queries: Vec<String> = vec![];
    // if a library module need to walk through gromet differently
    if gromet.modules[0].r#fn.bf.is_none() {
        queries.append(&mut create_function_net_lib(gromet, start));
    } else {
        // if executable code
        queries.append(&mut create_function_net(gromet, start));
    }
    queries
}

// This creates the graph queries from a function network if the code is not executable
// currently only supports creating the first attribute (a function) and all its dependencies
// need to add support to find next function and create network as well and repeat
#[allow(unused_assignments)]
fn create_function_net_lib(gromet: &ModuleCollection, mut start: u32) -> Vec<String> {
    let mut queries: Vec<String> = vec![];
    let mut nodes: Vec<Node> = vec![];
    let mut meta_nodes: Vec<MetadataNode> = vec![];
    let _metadata_idx = 0;
    let mut edges: Vec<Edge> = vec![];
    // in order to have less repetition for multiple function calls and to setup support for recursive functions
    // We check if the function node and thus contents were already made, and not duplicate the contents if already made
    let bf_counter: u8 = 1;
    let mut function_call_repeat = false;
    let mut original_bf = bf_counter;
    let _boxf = gromet.modules[0].attributes[0].clone();
    for node in nodes.clone() {
        if (1 == node.contents) && (node.n_type == "Function") {
            function_call_repeat = true;
            if node.nbox < original_bf as usize {
                original_bf = node.nbox as u8; // This grabs the first instance of bf that the function was called
                                               // and thus is the nbox value of the nodes of the original contents
            }
        }
    }
    if function_call_repeat {
        // This means the function has been called before so we don't fully construct the graph
        // constructing metadata node if metadata exists
        let mut metadata_idx = 0;
        let eboxf = gromet.modules[0].attributes[0_usize].clone();
        let n1 = Node {
            n_type: String::from("Function"),
            value: None,
            name: Some(
                eboxf.b.as_ref().unwrap()[0]
                    .name
                    .clone()
                    .map_or_else(|| format!("Function{}", start), |x| x),
            ),
            node_id: format!("n{}", start),
            out_idx: None,
            in_indx: None,
            contents: 1,
            nbox: bf_counter as usize,
            att_bf_idx: 0,
            box_counter: 0,
        };
        let e1 = Edge {
            src: String::from("mod"),
            tgt: format!("n{}", start),
            e_type: String::from("Contains"),
            prop: Some(1),
        };
        nodes.push(n1.clone());
        edges.push(e1);
        if eboxf.b.as_ref().unwrap()[0].metadata.as_ref().is_some() {
            metadata_idx = eboxf.b.unwrap()[0].metadata.unwrap();
            let mut repeat_meta = false;
            for node in meta_nodes.iter() {
                if node.metadata_idx == metadata_idx {
                    repeat_meta = true;
                }
            }
            if !repeat_meta {
                meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
        let eboxf = gromet.modules[0].attributes[idx].clone();

        let c_args = ConstructorArgs {
            att_box: eboxf.clone(),
            cur_box: eboxf.bf.unwrap()[0].clone(),
            parent_node: n1.clone(),
            att_idx: 1,
            bf_counter: 1,
            att_bf_idx: 0,
            box_counter: 0,
        };

        // create the ports
        create_opo(
            gromet,     // gromet for metadata
            &mut nodes, // nodes
            &mut edges,
            &mut meta_nodes,
            &mut start,
            c_args.clone(),
        );
        create_opi(
            gromet,     // gromet for metadata
            &mut nodes, // nodes
            &mut edges,
            &mut meta_nodes,
            &mut start,
            c_args.clone(),
        );
        // now to add the contains wires for the additional function call onto the original contents nodes:
        for node in nodes.clone() {
            if (node.nbox == original_bf as usize)
                && (node.contents == (idx + 1))
                && ((node.n_type == "Literal")
                    || (node.n_type == "Primitive")
                    || (node.n_type == "Predicate")
                    || (node.n_type == "Expression"))
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

        // now we need to wire these ports to the content nodes which already exist.
        // they will have the same contents, being: (idx+1), however the bf_counter will be different, parse bf_counter from first call
        // (smallest of bf_counter of all calls) and use that in wiring, it is original_bf now
        // concerns over wiring into an expression, the expression would be in the correct contents attribute, but the ports are labeled as the expressions contents
        for wire in eboxf.wfopi.unwrap().iter() {
            let mut wfopi_src_tgt: Vec<String> = vec![];
            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if original_bf == node.nbox as u8 {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if (idx + 1) == node.contents {
                        // only include nodes with pifs
                        if node.in_indx.is_some() {
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
                if bf_counter == node.nbox as u8 {
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
        for wire in eboxf.wfopo.unwrap().iter() {
            let mut wfopo_src_tgt: Vec<String> = vec![];
            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if bf_counter == node.nbox as u8 {
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
                if original_bf == node.nbox as u8 {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if (idx + 1) == node.contents {
                        // only include nodes with pofs
                        if node.out_idx.is_some() {
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
        let att_box = gromet.modules[0].attributes[0_usize].clone();

        let temp_node = Node {
            n_type: String::from("Temp"),
            value: None,
            name: None,
            node_id: "mod".to_string(),
            out_idx: None,
            in_indx: None,
            contents: 0,
            nbox: 0,
            att_bf_idx: 0,
            box_counter: 0,
        };

        // this represents the default state for this struct
        let c_args = ConstructorArgs {
            att_box: att_box.clone(),
            cur_box: att_box.bf.unwrap()[0].clone(),
            parent_node: temp_node,
            att_idx: 1,
            bf_counter: 0,
            att_bf_idx: 0,
            box_counter: 0,
        };

        create_function(
            gromet,     // gromet for metadata
            &mut nodes, // nodes
            &mut edges,
            &mut meta_nodes,
            &mut start,
            c_args.clone(),
        );
    }
    // add wires for inbetween attribute level boxes, so opo's, opi's and module level literals
    // between attributes
    // get wired through module level wff field, will require reading through node list to
    // match contents field to box field on wff entries
    external_wiring(gromet, &mut nodes.clone(), &mut edges);

    // make conditionals if they exist
    /* I am commenteding out conditionals and loops for Library calls as we get duplicated traverals for now (10/18/23) */
    /*if gromet.modules[0].r#fn.bc.as_ref().is_some() {
        let mut cond_counter = 0;
        let temp_mod_node = Node {
            n_type: String::from("module"),
            value: None,
            name: None, // I think this naming will get messed up if there are multiple ports...
            node_id: "mod".to_string(),
            out_idx: None,
            in_indx: None,
            contents: 0,
            nbox: 0,
            att_bf_idx: 0,
            box_counter: 0,
        };
        let c_args = ConstructorArgs {
            att_box: gromet.modules[0].r#fn.clone(),
            cur_box: gromet.modules[0].attributes[0].clone().bf.unwrap()[0].clone(), // this is a temp fill
            parent_node: temp_mod_node,
            att_idx: 1,
            bf_counter: 0,
            att_bf_idx: 0,
            box_counter: 0,
        };
        for _cond in gromet.modules[0].r#fn.bc.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            create_conditional(
                gromet,     // gromet for metadata
                &mut nodes, // nodes
                &mut edges,
                &mut meta_nodes,
                &mut start,
                c_args.clone(),
                cond_counter,
            );
            cond_counter += 1;
        }
    }
    // make conditionals if they exist
    if gromet.modules[0].r#fn.bl.as_ref().is_some() {
        let mut while_counter = 0;
        let temp_mod_node = Node {
            n_type: String::from("module"),
            value: None,
            name: None, // I think this naming will get messed up if there are multiple ports...
            node_id: "mod".to_string(),
            out_idx: None,
            in_indx: None,
            contents: 0,
            nbox: 0,
            att_bf_idx: 0,
            box_counter: 0,
        };
        let c_args = ConstructorArgs {
            att_box: gromet.modules[0].r#fn.clone(),
            cur_box: gromet.modules[0].attributes[0].clone().bf.unwrap()[0].clone(),
            parent_node: temp_mod_node,
            att_idx: 1,
            bf_counter: 0,
            att_bf_idx: 0,
            box_counter: 0,
        };
        for _while_l in gromet.modules[0].r#fn.bl.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            if gromet.modules[0].r#fn.bl.as_ref().unwrap()[while_counter as usize]
                .pre
                .is_none()
            {
                // if there is not a pre_condition then it is a while loop
                create_while_loop(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args.clone(),
                    while_counter,
                );
            } else {
                // if there is pre condition then it is a for loop
                create_for_loop(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args.clone(),
                    while_counter,
                );
            }
            while_counter += 1;
        }
    } */
    // convert every node object into a node query
    let create = String::from("CREATE");
    for node in nodes.iter() {
        let mut name = String::from("a");
        if node.name.is_none() {
            name = node.n_type.clone();
        } else {
            name = node.name.as_ref().unwrap().to_string();
        }
        let value = match &node.value {
            Some(val) => format!(
                "{{ value_type:{:?}, value:{:?}, gromet_type:{:?} }}",
                val.value_type,
                val.value,
                val.gromet_type.as_ref().unwrap()
            ),
            None => String::from("\"\""),
        };

        // NOTE: The format of value has changed to represent a literal Cypher map {field:value}.
        // We no longer need to format value with the debug :? parameter
        let node_query = format!(
            "{} ({}:{} {{name:{:?},value:{},order_box:{:?},order_att:{:?}}})",
            create, node.node_id, node.n_type, name, value, node.nbox, node.contents
        );
        queries.push(node_query);
    }
    for node in meta_nodes.iter() {
        queries.append(&mut create_metadata_node_query(node.clone()));
    }

    // convert every edge object into an edge query
    let init_edges = edges.len();
    edges.sort();
    edges.dedup();
    let edges_clone = edges.clone();
    // also dedup if edge prop is different
    for (i, edge) in edges_clone.iter().enumerate().rev() {
        if i != 0 {
            if edge.src == edges_clone[i-1].src && edge.tgt == edges_clone[i-1].tgt {
                edges.remove(i);
            }
        }
    }
    let fin_edges = edges.len();
    if init_edges != fin_edges {
        println!("Duplicated Edges Removed, check for bugs");
    }
    for edge in edges.iter() {
        let edge_query = format!(
            "{} ({})-[e{}{}:{}]->({})",
            create, edge.src, edge.src, edge.tgt, edge.e_type, edge.tgt
        );
        queries.push(edge_query);

        if edge.prop.is_some() {
            let set_query = format!("set e{}{}.index={}", edge.src, edge.tgt, edge.prop.unwrap());
            queries.push(set_query);
        }
    }
    queries
}

#[allow(unused_assignments)]
fn create_function_net(gromet: &ModuleCollection, mut start: u32) -> Vec<String> {
    // intialize the vectors
    let mut queries: Vec<String> = vec![];
    let mut nodes: Vec<Node> = vec![];
    let mut meta_nodes: Vec<MetadataNode> = vec![];
    let mut metadata_idx = 0;
    let mut edges: Vec<Edge> = vec![];

    // temp node for calling the module as parent for top level
    let temp_node = Node {
        n_type: String::from("Temp"),
        value: None,
        name: None,
        node_id: "mod".to_string(),
        out_idx: None,
        in_indx: None,
        contents: 0,
        nbox: 0,
        att_bf_idx: 0,
        box_counter: 0,
    };

    // this represents the default state for this struct
    let mut c_args = ConstructorArgs {
        att_box: gromet.modules[0].r#fn.clone(),
        cur_box: gromet.modules[0].r#fn.bf.as_ref().unwrap()[0].clone(),
        parent_node: temp_node.clone(),
        att_idx: 0,
        bf_counter: 0,
        att_bf_idx: 0,
        box_counter: 0,
    };

    let mut bf_counter: usize = 1;
    for boxf in gromet.modules[0].r#fn.bf.as_ref().unwrap().iter() {
        c_args.bf_counter = bf_counter;
        c_args.cur_box = boxf.clone();
        c_args.att_idx = 0; // reset incase we hit a function and then literal for example
                            // construct the sub module level boxes along with their metadata and connection to module
        c_args.parent_node = temp_node.clone(); // for if overwritten in function constructor
        c_args.att_box = gromet.modules[0].r#fn.clone(); // incase over written
        match boxf.function_type {
            FunctionType::Primitive => {
                create_att_primitive(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args.clone(),
                );
            }
            FunctionType::Literal => {
                create_att_literal(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args.clone(),
                );
            }
            FunctionType::Predicate => {
                c_args.att_idx = boxf.contents.unwrap() as usize;
                c_args.att_box = gromet.modules[0].attributes[c_args.att_idx - 1].clone();
                create_att_predicate(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args.clone(),
                );
            }
            FunctionType::Expression => {
                c_args.att_idx = boxf.contents.unwrap() as usize;
                c_args.att_box = gromet.modules[0].attributes[c_args.att_idx - 1].clone();
                create_att_expression(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args.clone(),
                );
            }
            FunctionType::Function => {
                // in order to have less repetition for multiple function calls and to setup support for recursive functions
                // We check if the function node and thus contents were already made, and not duplicate the contents if already made
                c_args.att_idx = boxf.contents.unwrap() as usize;
                c_args.att_box = gromet.modules[0].attributes[c_args.att_idx - 1].clone();
                let mut function_call_repeat = false;
                let mut original_bf = bf_counter;
                for node in nodes.clone() {
                    if (boxf.contents.unwrap() == node.contents as u32)
                        && (node.n_type == "Function")
                    {
                        function_call_repeat = true;
                        if node.nbox < original_bf {
                            original_bf = node.nbox; // This grabs the first instance of bf that the function was called
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
                            eboxf.b.as_ref().unwrap()[0]
                                .name
                                .clone()
                                .map_or_else(|| format!("Function{}", start), |x| x),
                        ),
                        node_id: format!("n{}", start),
                        out_idx: None,
                        in_indx: None,
                        contents: idx as usize + 1,
                        nbox: bf_counter,
                        att_bf_idx: 0,
                        box_counter: 0,
                    };
                    let e1 = Edge {
                        src: String::from("mod"),
                        tgt: format!("n{}", start),
                        e_type: String::from("Contains"),
                        prop: Some(boxf.contents.unwrap() as usize),
                    };
                    nodes.push(n1.clone());
                    edges.push(e1);
                    // bf level metadata reference
                    if boxf.metadata.as_ref().is_some() {
                        metadata_idx = boxf.metadata.unwrap();
                        let mut repeat_meta = false;
                        for node in meta_nodes.iter() {
                            if node.metadata_idx == metadata_idx {
                                repeat_meta = true;
                            }
                        }
                        if !repeat_meta {
                            meta_nodes
                                .append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
                    if eboxf.b.as_ref().unwrap()[0].metadata.as_ref().is_some() {
                        metadata_idx = eboxf.b.unwrap()[0].metadata.unwrap();
                        let mut repeat_meta = false;
                        for node in meta_nodes.iter() {
                            if node.metadata_idx == metadata_idx {
                                repeat_meta = true;
                            }
                        }
                        if !repeat_meta {
                            meta_nodes
                                .append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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

                    c_args.parent_node = n1.clone();

                    // construct opo nodes, if not none
                    create_opo(
                        gromet,     // gromet for metadata
                        &mut nodes, // nodes
                        &mut edges,
                        &mut meta_nodes,
                        &mut start,
                        c_args.clone(),
                    );
                    create_opi(
                        gromet,     // gromet for metadata
                        &mut nodes, // nodes
                        &mut edges,
                        &mut meta_nodes,
                        &mut start,
                        c_args.clone(),
                    );
                    // now to add the contains wires for the additional function call onto the original contents nodes:
                    for node in nodes.clone() {
                        if (node.nbox == original_bf)
                            && (node.contents == (idx as usize + 1))
                            && ((node.n_type == "Literal")
                                || (node.n_type == "Primitive")
                                || (node.n_type == "Predicate")
                                || (node.n_type == "Expression"))
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

                    // now we need to wire these ports to the content nodes which already exist.
                    // they will have the same contents, being: (idx+1), however the bf_counter will be different, parse bf_counter from first call
                    // (smallest of bf_counter of all calls) and use that in wiring, it is original_bf now
                    // concerns over wiring into an expression, the expression would be in the correct contents attribute, but the ports are labeled as the expressions contents
                    for wire in eboxf.wfopi.unwrap().iter() {
                        let mut wfopi_src_tgt: Vec<String> = vec![];
                        // find the src node
                        for node in nodes.iter() {
                            // make sure in correct box
                            if original_bf == node.nbox {
                                // make sure only looking in current attribute nodes for srcs and tgts
                                if (idx + 1) == node.contents as u32 {
                                    // only include nodes with pifs
                                    if node.in_indx.is_some() {
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
                                if (idx + 1) == node.contents as u32 {
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
                    for wire in eboxf.wfopo.unwrap().iter() {
                        let mut wfopo_src_tgt: Vec<String> = vec![];
                        // find the src node
                        for node in nodes.iter() {
                            // make sure in correct box
                            if bf_counter == node.nbox {
                                // make sure only looking in current attribute nodes for srcs and tgts
                                if (idx + 1) == node.contents as u32 {
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
                                if (idx + 1) == node.contents as u32 {
                                    // only include nodes with pofs
                                    if node.out_idx.is_some() {
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
                    // new function call
                    c_args.att_idx = boxf.contents.unwrap() as usize;
                    create_function(
                        gromet,     // gromet for metadata
                        &mut nodes, // nodes
                        &mut edges,
                        &mut meta_nodes,
                        &mut start,
                        c_args.clone(),
                    );
                }
            }
            FunctionType::Imported => {
                create_import(gromet, &mut nodes, &mut edges, &mut meta_nodes, &mut start, c_args.clone());
                start += 1;
                // now to implement wiring
                import_wiring(
                    &gromet.clone(),
                    &mut nodes,
                    &mut edges,
                    c_args.att_idx,
                    c_args.bf_counter,
                    c_args.parent_node.clone(),
                );
            }
            FunctionType::ImportedMethod => {
                // basically seems like these are just functions to me. 
                c_args.att_idx = boxf.contents.unwrap() as usize;
                c_args.att_box = gromet.modules[0].attributes[c_args.att_idx - 1].clone();
                create_function(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args.clone(),
                );
            }
            _ => {}
        }
        start += 1;
        bf_counter += 1;
    }
    start += 1;

    // add wires for inbetween attribute level boxes, so opo's, opi's and module level literals
    // between attributes
    // get wired through module level wff field, will require reading through node list to
    // match contents field to box field on wff entries
    external_wiring(gromet, &mut nodes.clone(), &mut edges);

    let c_args_other = ConstructorArgs {
        att_box: gromet.modules[0].r#fn.clone(),
        cur_box: gromet.modules[0].r#fn.bf.as_ref().unwrap()[0].clone(),
        parent_node: temp_node.clone(),
        att_idx: 0,
        bf_counter: 0,
        att_bf_idx: 0,
        box_counter: 0,
    };

    // make conditionals if they exist
    if gromet.modules[0].r#fn.bc.as_ref().is_some() {
        let mut cond_counter = 0;
        for _cond in gromet.modules[0].r#fn.bc.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            create_conditional(
                gromet,     // gromet for metadata
                &mut nodes, // nodes
                &mut edges,
                &mut meta_nodes,
                &mut start,
                c_args_other.clone(),
                cond_counter,
            );
            cond_counter += 1;
        }
    }
    // make loops if they exist
    if gromet.modules[0].r#fn.bl.as_ref().is_some() {
        let mut while_counter = 0;
        for _loop_count in gromet.modules[0].r#fn.bl.as_ref().unwrap().iter() {
            if gromet.modules[0].r#fn.bl.as_ref().unwrap()[while_counter as usize]
                .pre
                .is_none()
            {
                // if there is not a pre_condition then it is a while loop
                create_while_loop(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args_other.clone(),
                    while_counter,
                );
            } else {
                // if there is pre condition then it is a for loop
                create_for_loop(
                    gromet,     // gromet for metadata
                    &mut nodes, // nodes
                    &mut edges,
                    &mut meta_nodes,
                    &mut start,
                    c_args_other.clone(),
                    while_counter,
                );
            }
            while_counter += 1;
        }
    }
    // convert every node object into a node query
    let create = String::from("CREATE");
    for node in nodes.iter() {
        let mut name = String::from("a");
        if node.name.is_none() {
            name = node.n_type.clone();
        } else {
            name = node.name.as_ref().unwrap().to_string();
        }
        let value = match &node.value {
            Some(val) => format!(
                "{{ value_type:{:?}, value:{:?}, gromet_type:{:?} }}",
                val.value_type,
                val.value,
                val.gromet_type.as_ref().unwrap()
            ),
            None => String::from("\"\""),
        };

        // NOTE: The format of value has changed to represent a literal Cypher map {field:value}.
        // We no longer need to format value with the debug :? parameter
        let node_query = format!(
            "{} ({}:{} {{name:{:?},value:{},order_box:{:?},order_att:{:?}}})",
            create, node.node_id, node.n_type, name, value, node.nbox, node.contents
        );
        queries.push(node_query);
    }
    for node in meta_nodes.iter() {
        queries.append(&mut create_metadata_node_query(node.clone()));
    }

    // convert every edge object into an edge query
    let init_edges = edges.len();
    edges.sort();
    edges.dedup();
    let edges_clone = edges.clone();
    // also dedup if edge prop is different
    for (i, edge) in edges_clone.iter().enumerate().rev() {
        if i != 0 {
            if edge.src == edges_clone[i-1].src && edge.tgt == edges_clone[i-1].tgt {
                edges.remove(i);
            }
        }
    }
    let fin_edges = edges.len();
    if init_edges != fin_edges {
        println!("Duplicated Edges Removed, check for bugs");
    }
    for edge in edges.iter() {
        let edge_query = format!(
            "{} ({})-[e{}{}:{}]->({})",
            create, edge.src, edge.src, edge.tgt, edge.e_type, edge.tgt
        );
        queries.push(edge_query);

        if edge.prop.is_some() {
            let set_query = format!("set e{}{}.index={}", edge.src, edge.tgt, edge.prop.unwrap());
            queries.push(set_query);
        }
    }
    queries
}
// this method creates an import type function
// currently assumes top level call
#[allow(unused_assignments)]
pub fn create_import(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    let eboxf = gromet.modules[0].clone();
    let mut sboxf = gromet.modules[0].r#fn.clone(); 
    if c_args.att_idx != 0 {
        sboxf = gromet.modules[0].attributes[c_args.att_idx - 1].clone();
    }
    //let mboxf = eboxf.r#fn.bf.unwrap()[c_args.bf_counter - 1].clone();

    let mut pof: Vec<u32> = vec![];
    if eboxf.r#fn.pof.is_some() {
        let mut po_idx: u32 = 1;
        for port in eboxf.r#fn.pof.clone().unwrap().iter() {
            if port.r#box == c_args.bf_counter as u8 {
                pof.push(po_idx);
            }
            po_idx += 1;
        }
    }
    // then find pif's for box
    let mut pif: Vec<u32> = vec![];
    if eboxf.r#fn.pif.is_some() {
        let mut pi_idx: u32 = 1;
        for port in eboxf.r#fn.pif.unwrap().iter() {
            if port.r#box == c_args.bf_counter as u8 {
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
        name: sboxf.name,
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: Some(pif),
        contents: c_args.att_idx,
        nbox: c_args.bf_counter,
        att_bf_idx: c_args.att_bf_idx,
        box_counter: c_args.box_counter,
    };
    nodes.push(n3.clone());
    // make edge connecting to expression
    let e4 = Edge {
        src: c_args.parent_node.node_id,
        tgt: n3.node_id.clone(),
        e_type: String::from("Contains"),
        prop: None,
    };
    edges.push(e4);
    if eboxf.metadata.is_some() {
        metadata_idx = eboxf.metadata.unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
            let me1 = Edge {
                src: n3.node_id,
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }
    *start += 1;
}

// this creates a function node including all the contents included in it, including additional functions
// CHANGEs: removed function_net arg and added box_counter arg
#[allow(unused_assignments)]
pub fn create_function(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    // function is not repeated
    let att_box = gromet.modules[0].attributes[c_args.att_idx - 1].clone(); // current expression attribute
    let mut parent_node = c_args.parent_node.clone();
    // now we add a check for if this is an imported function
    match att_box.b.clone().unwrap()[0].function_type.clone() {
        FunctionType::Imported => {
            create_import(gromet, nodes, edges, meta_nodes, start, c_args.clone());
            *start += 1;
            // now to implement wiring
            import_wiring(
                &gromet.clone(),
                nodes,
                edges,
                c_args.att_idx,
                c_args.bf_counter,
                c_args.parent_node.clone(),
            );
        }
        _ => {
            if att_box.bf.is_some() {
                let n1 = Node {
                    n_type: String::from("Function"),
                    value: None,
                    name: Some(
                        att_box.b.as_ref().unwrap()[0]
                            .name
                            .clone()
                            .map_or_else(|| format!("Function{}", start), |x| x),
                    ),
                    node_id: format!("n{}", start),
                    out_idx: None,
                    in_indx: None,
                    contents: c_args.att_idx,
                    nbox: c_args.bf_counter,
                    att_bf_idx: c_args.att_bf_idx,
                    box_counter: c_args.box_counter,
                };
                let e1 = Edge {
                    src: c_args.parent_node.node_id.clone(),
                    tgt: n1.node_id.clone(),
                    e_type: String::from("Contains"),
                    prop: Some(c_args.att_idx),
                };
                parent_node = n1.clone();
                nodes.push(n1.clone());
                edges.push(e1);
                let mut metadata_idx = 0;
                // attribute b level metadata reference
                if att_box.b.as_ref().unwrap()[0].metadata.as_ref().is_some() {
                    metadata_idx = att_box.b.as_ref().unwrap()[0].metadata.unwrap();
                    let mut repeat_meta = false;
                    for node in meta_nodes.iter() {
                        if node.metadata_idx == metadata_idx {
                            repeat_meta = true;
                        }
                    }
                    if !repeat_meta {
                        meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
                *start += 1;

                let mut new_c_args = c_args.clone();
                new_c_args.parent_node = n1.clone();
                new_c_args.att_bf_idx = c_args.att_idx; // This is the parent index

                // construct opo nodes, if not none, might need to
                create_opo(
                    gromet, // gromet for metadata
                    nodes,  // nodes
                    edges,
                    meta_nodes,
                    start,
                    new_c_args.clone(),
                );
                create_opi(
                    gromet, // gromet for metadata
                    nodes,  // nodes
                    edges,
                    meta_nodes,
                    start,
                    new_c_args.clone(),
                );
                // now to construct the nodes inside the function, currently supported Literals and Primitives
                // first include an Expression for increased depth
                let mut box_counter: usize = 1;
                for att_sub_box in att_box.bf.as_ref().unwrap().iter() {
                    new_c_args.box_counter = box_counter;
                    new_c_args.cur_box = att_sub_box.clone();
                    new_c_args.att_idx = c_args.att_idx;
                    match att_sub_box.function_type {
                        FunctionType::Function => {
                            new_c_args.att_idx = att_sub_box.contents.unwrap() as usize;
                            create_function(
                                gromet, // gromet for metadata
                                nodes,  // nodes
                                edges,
                                meta_nodes,
                                start,
                                new_c_args.clone(),
                            );
                        }

                        FunctionType::Predicate => {
                            new_c_args.att_idx = att_sub_box.contents.unwrap() as usize;
                            create_att_predicate(
                                gromet, // gromet for metadata
                                nodes,  // nodes
                                edges,
                                meta_nodes,
                                start,
                                new_c_args.clone(),
                            );
                        }
                        FunctionType::Expression => {
                            new_c_args.att_idx = att_sub_box.contents.unwrap() as usize;
                            create_att_expression(
                                gromet, // gromet for metadata
                                nodes,  // nodes
                                edges,
                                meta_nodes,
                                start,
                                new_c_args.clone(),
                            );
                        }
                        FunctionType::Literal => {
                            create_att_literal(
                                gromet, // gromet for metadata
                                nodes,  // nodes
                                edges,
                                meta_nodes,
                                start,
                                new_c_args.clone(),
                            );
                        }
                        FunctionType::Primitive => {
                            create_att_primitive(
                                gromet, // gromet for metadata
                                nodes,  // nodes
                                edges,
                                meta_nodes,
                                start,
                                new_c_args.clone(),
                            );
                        }
                        FunctionType::Abstract => {
                            create_att_abstract(
                                gromet, // gromet for metadata
                                nodes,  // nodes
                                edges,
                                meta_nodes,
                                start,
                                new_c_args.clone(),
                            );
                        }
                        FunctionType::ImportedMethod => {
                            // this is a function call, but for some reason is not called a function
                            new_c_args.att_idx = att_sub_box.contents.unwrap() as usize;
                            create_function(
                                gromet, // gromet for metadata
                                nodes,  // nodes
                                edges,
                                meta_nodes,
                                start,
                                new_c_args.clone(),
                            );
                        }
                        FunctionType::Imported => {
                            create_import(gromet, nodes, edges, meta_nodes, start, c_args.clone());
                            *start += 1;
                            // now to implement wiring
                            import_wiring(
                                &gromet.clone(),
                                nodes,
                                edges,
                                c_args.att_idx,
                                c_args.bf_counter,
                                c_args.parent_node.clone(),
                            );
                        }
                        _ => {
                            println!(
                                "Missing a box in a function! {:?}",
                                att_sub_box.function_type.clone()
                            );
                        }
                    }
                    box_counter += 1;
                    *start += 1;
                }

                // Now we perform the internal wiring of this branch
                internal_wiring(
                    att_box.clone(),
                    nodes,
                    edges,
                    c_args.att_idx,
                    c_args.bf_counter,
                );
                // perform cross attributal wiring of function
                cross_att_wiring(
                    att_box.clone(),
                    nodes,
                    edges,
                    c_args.att_idx,
                    c_args.bf_counter,
                );
            } else {
                create_import(gromet, nodes, edges, meta_nodes, start, c_args.clone());
                *start += 1;
                // now to implement wiring
                import_wiring(
                    &gromet.clone(),
                    nodes,
                    edges,
                    c_args.att_idx,
                    c_args.bf_counter,
                    c_args.parent_node.clone(),
                );
            }
        }
    }
    let mut new_c_args = c_args.clone();
    new_c_args.parent_node = parent_node.clone();
    // make conditionals if they exist
    if att_box.bc.as_ref().is_some() {
        let mut cond_counter = 0;
        for _cond in att_box.bc.as_ref().unwrap().iter() {
            // now lets check for and setup any conditionals at this level
            create_conditional(
                gromet, // gromet for metadata
                nodes,  // nodes
                edges,
                meta_nodes,
                start,
                new_c_args.clone(),
                cond_counter,
            );
            cond_counter += 1;
        }
    }
    // make loops if they exist
    if att_box.bl.as_ref().is_some() {
        let mut while_counter = 0;
        for _loop_count in att_box.bl.as_ref().unwrap().iter() {
            if att_box.bl.as_ref().unwrap()[while_counter as usize]
                .pre
                .is_none()
            {
                // if there is not a pre_condition then it is a while loop
                create_while_loop(
                    gromet, // gromet for metadata
                    nodes,  // nodes
                    edges,
                    meta_nodes,
                    start,
                    new_c_args.clone(),
                    while_counter,
                );
            } else {
                // if there is pre condition then it is a for loop
                create_for_loop(
                    gromet, // gromet for metadata
                    nodes,  // nodes
                    edges,
                    meta_nodes,
                    start,
                    new_c_args.clone(),
                    while_counter,
                );
            }
            while_counter += 1;
        }
    }
}

// this creates the framework for conditionals, including the conditional node, the pic and poc
// nodes and the cond, body_if and body_else edges
// The iterator through the conditionals will need to be outside this function
#[allow(unused_assignments)]
pub fn create_conditional(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
    cond_counter: u32,
) {
    let bf_counter = c_args.bf_counter;
    let att_idx = c_args.att_idx;
    let att_bf_idx = c_args.att_bf_idx;
    let box_counter = c_args.box_counter;
    let function_net = c_args.att_box.clone();

    let mut metadata_idx = 0;
    let n1 = Node {
        n_type: String::from("Conditional"),
        value: None,
        name: Some(format!("Conditional{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: att_idx,
        nbox: bf_counter,
        att_bf_idx,
        box_counter,
    };
    let e1 = Edge {
        src: c_args.parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(cond_counter as usize),
    };
    nodes.push(n1.clone());
    edges.push(e1);

    *start += 1;

    // now we make the pic and poc ports and connect them to the conditional node
    if function_net.pic.is_some() {
        let mut port_count = 1;
        for pic in function_net.pic.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut pic_name = String::from("Pic");
            if pic.name.as_ref().is_some() {
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
                contents: att_idx,
                nbox: bf_counter,
                att_bf_idx: 0,
                box_counter: 0,
            };
            let e3 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n2.clone());
            edges.push(e3);
            if pic.metadata.as_ref().is_some() {
                metadata_idx = pic.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
        }
    }
    if function_net.poc.is_some() {
        let mut port_count = 1;
        for poc in function_net.poc.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut poc_name = String::from("Poc");
            if poc.name.as_ref().is_some() {
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
                contents: att_idx,
                nbox: bf_counter,
                att_bf_idx: 0,
                box_counter: 0,
            };
            let e5 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n3.clone());
            edges.push(e5);
            if poc.metadata.as_ref().is_some() {
                metadata_idx = poc.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
        }
    }

    // Now let's create the edges for the conditional condition and it's bodies
    // find the nodes
    // the contents is the node's attribute reference, so need to pull off of box's contents value
    let cond_box = function_net.bc.as_ref().unwrap()[cond_counter as usize]
        .condition
        .unwrap();
    let body_if_box = function_net.bc.as_ref().unwrap()[cond_counter as usize]
        .body_if
        .unwrap();
    let mut body_else_box = body_if_box.clone();
    let mut else_exists = false;
    if function_net.bc.as_ref().unwrap()[cond_counter as usize]
        .body_else
        .is_some()
    {
        body_else_box = function_net.bc.as_ref().unwrap()[cond_counter as usize]
            .body_else
            .unwrap();
        else_exists = true;
    }

    // now we make the conditional, if, and else nodes

    let cond_att_box = gromet.modules[0].attributes[(cond_box - 1) as usize].clone();
    let if_att_box = gromet.modules[0].attributes[(body_if_box - 1) as usize].clone();
    let mut else_att_box = if_att_box.clone();
    if else_exists {
        else_att_box = gromet.modules[0].attributes[(body_else_box - 1) as usize].clone();
    }

    let mut new_c_args = c_args.clone();
    new_c_args.parent_node = n1.clone();
    // creating the conditional
    new_c_args.att_box = cond_att_box.clone();
    new_c_args.att_idx = cond_box as usize;
    create_att_predicate(gromet, nodes, edges, meta_nodes, start, new_c_args.clone());
    *start += 1;

    for edge in edges.iter_mut() {
        if edge.src == new_c_args.parent_node.node_id.clone() && edge.e_type == *"Contains" {
            edge.e_type = "Condition".to_string();
        }
    }

    // creating the if body
    new_c_args.att_box = if_att_box.clone();
    new_c_args.att_idx = body_if_box as usize;
    create_function(gromet, nodes, edges, meta_nodes, start, new_c_args.clone());
    *start += 1;

    for edge in edges.iter_mut() {
        if edge.src == new_c_args.parent_node.node_id.clone() && edge.e_type == *"Contains" {
            edge.e_type = "if_body".to_string();
        }
    }

    // creating the else body
    new_c_args.att_box = else_att_box.clone();
    new_c_args.att_idx = body_else_box as usize;
    create_function(gromet, nodes, edges, meta_nodes, start, new_c_args.clone());
    *start += 1;

    for edge in edges.iter_mut() {
        if edge.src == new_c_args.parent_node.node_id.clone() && edge.e_type == *"Contains" {
            edge.e_type = "else_body".to_string();
        }
    }

    // Now we start to wire these objects together there are two unique wire types and implicit wires that need to be made
    if function_net.wfc.is_some() {
        for wire in function_net.wfc.as_ref().unwrap().iter() {
            // collect info to identify the opi src node
            let src_idx = wire.src; // port index
            let src_att = att_idx; // attribute index of submodule (also opi contents value)
            let src_nbox = bf_counter; // nbox value of src opi
            let src_pic_idx = src_idx;

            let tgt_idx = wire.tgt; // port index
            let tgt_pof = function_net.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
            let tgt_opo_idx = tgt_pof.id.unwrap(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
            let tgt_box = tgt_pof.r#box; // tgt sub module box number

            let mut tgt_att = src_att;
            if function_net.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .contents
                .is_some()
            {
                tgt_att = function_net.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                    .contents
                    .unwrap() as usize; // attribute index of submodule (also opo contents value)
            }
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
                if tgt_nbox == node.nbox as u8 {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att as u32 == node.contents as u32 {
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
    }

    // now to make the implicit wires that go from pics -> /opis and /opos -> pocs.
    // every opi in predicate, if and else statements gets mapped to a pic of the same id
    // to determine the opi we need, att_idx, bf_counter, and id
    // to determine the pic we only need the id
    for opi in cond_att_box.opi.unwrap().iter() {
        let src_id = opi.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = cond_box as usize;

        let mut cond_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            if node.n_type == "Pic" {
                // iterate through port to check for tgt
                for p in node.in_indx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (src_id as u32) == *p {
                        cond_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_id as u32) == *p {
                                cond_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if cond_src_tgt.len() == 2 {
            let e9 = Edge {
                src: cond_src_tgt[0].clone(),
                tgt: cond_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e9);
        }
    }
    for opi in if_att_box.opi.unwrap().iter() {
        let src_id = opi.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = body_if_box as usize;

        let mut if_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            if node.n_type == "Pic" {
                // iterate through port to check for tgt
                for p in node.in_indx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (src_id as u32) == *p {
                        if_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_id as u32) == *p {
                                if_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if if_src_tgt.len() == 2 {
            let e10 = Edge {
                src: if_src_tgt[0].clone(),
                tgt: if_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e10);
        }
    }
    if else_exists {
        for opi in else_att_box.opi.unwrap().iter() {
            let src_id = opi.id.unwrap();
            let tgt_id = src_id;
            let tgt_bf_counter = 0; // because of how we have defined it
            let tgt_att_idx = body_else_box as usize;

            let mut else_src_tgt: Vec<String> = vec![];

            for node in nodes.iter() {
                if node.n_type == "Pic" {
                    // iterate through port to check for tgt
                    for p in node.in_indx.as_ref().unwrap().iter() {
                        // push the src first, being pif
                        if (src_id as u32) == *p {
                            else_src_tgt.push(node.node_id.clone());
                        }
                    }
                }
            }
            for node in nodes.iter() {
                // make sure in correct box
                if tgt_bf_counter == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att_idx == node.contents {
                        // only opo's
                        if node.n_type == "Opi" {
                            // iterate through port to check for tgt
                            for p in node.in_indx.as_ref().unwrap().iter() {
                                // push the src first, being pif
                                if (tgt_id as u32) == *p {
                                    else_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
            if else_src_tgt.len() == 2 {
                let e11 = Edge {
                    src: else_src_tgt[0].clone(),
                    tgt: else_src_tgt[1].clone(),
                    e_type: String::from("Wire"),
                    prop: None,
                };
                edges.push(e11);
            }
        }
    }

    // every opo in if and else statements gets mapped to a poc of the same id, every opo but the last in a predicate
    // gets mapped to a poc of the same id.

    for opo in if_att_box.opo.clone().unwrap().iter() {
        let src_id = opo.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = body_if_box as usize;

        let mut if_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_id as u32) == *p {
                                if_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        for node in nodes.iter() {
            if node.n_type == "Poc" {
                // iterate through port to check for tgt
                for p in node.out_idx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (tgt_id as u32) == *p {
                        if_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        if if_src_tgt.len() == 2 {
            let e12 = Edge {
                src: if_src_tgt[0].clone(),
                tgt: if_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e12);
        }
    }
    if else_exists {
        for opo in else_att_box.opo.unwrap().iter() {
            let src_id = opo.id.unwrap();
            let tgt_id = src_id;
            let tgt_bf_counter = 0; // because of how we have defined it
            let tgt_att_idx = body_else_box as usize;

            let mut else_src_tgt: Vec<String> = vec![];

            for node in nodes.iter() {
                // make sure in correct box
                if tgt_bf_counter == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att_idx == node.contents {
                        // only opo's
                        if node.n_type == "Opo" {
                            // iterate through port to check for tgt
                            for p in node.out_idx.as_ref().unwrap().iter() {
                                // push the src first, being pif
                                if (src_id as u32) == *p {
                                    else_src_tgt.push(node.node_id.clone());
                                }
                            }
                        }
                    }
                }
            }
            for node in nodes.iter() {
                if node.n_type == "Poc" {
                    // iterate through port to check for tgt
                    for p in node.out_idx.as_ref().unwrap().iter() {
                        // push the src first, being pif
                        if (tgt_id as u32) == *p {
                            else_src_tgt.push(node.node_id.clone());
                        }
                    }
                }
            }
            if else_src_tgt.len() == 2 {
                let e13 = Edge {
                    src: else_src_tgt[0].clone(),
                    tgt: else_src_tgt[1].clone(),
                    e_type: String::from("Wire"),
                    prop: None,
                };
                edges.push(e13);
            }
        }
    }
    // iterate through everything but last opo
    let opo_final_idx = cond_att_box.opo.clone().unwrap().len() - 1;
    for (i, opo) in cond_att_box.opo.unwrap().iter().enumerate() {
        let src_id = opo.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = cond_box as usize;

        let mut cond_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_id as u32) == *p {
                                cond_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        for node in nodes.iter() {
            if node.n_type == "Poc" {
                // iterate through port to check for tgt
                for p in node.out_idx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (tgt_id as u32) == *p {
                        cond_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        if opo_final_idx != i {
            if cond_src_tgt.len() == 2 {
                let e14 = Edge {
                    src: cond_src_tgt[0].clone(),
                    tgt: cond_src_tgt[1].clone(),
                    e_type: String::from("Wire"),
                    prop: None,
                };
                edges.push(e14);
            }
        }
    }
}
pub fn create_for_loop(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
    cond_counter: u32,
) {
    let bf_counter = c_args.bf_counter;
    let att_idx = c_args.att_idx;
    let _att_bf_idx = c_args.att_bf_idx;
    let _box_counter = c_args.box_counter;
    let function_net = c_args.att_box.clone();

    let mut metadata_idx = 0;
    let n1 = Node {
        n_type: String::from("For_Loop"),
        value: None,
        name: Some(format!("For{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: att_idx,
        nbox: bf_counter,
        att_bf_idx: 0,
        box_counter: 0,
    };
    let e1 = Edge {
        src: c_args.parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(cond_counter as usize),
    };
    nodes.push(n1.clone());
    edges.push(e1);
    if function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .metadata
        .as_ref()
        .is_some()
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
            meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
            let me1 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }

    *start += 1;

    // now we construct the condition and body of the loop, assumption: condition is predicate, and body if a function
    let pre_att = function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .pre
        .unwrap();
    let condition_att = function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .condition
        .unwrap();
    let body_att = function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .body
        .unwrap();

    let pre_att_box = gromet.modules[0].attributes[(pre_att - 1) as usize].clone();
    let cond_att_box = gromet.modules[0].attributes[(condition_att - 1) as usize].clone();
    let body_att_box = gromet.modules[0].attributes[(body_att - 1) as usize].clone();

    let mut new_c_args = c_args.clone();

    // first create the pre function
    new_c_args.parent_node = n1.clone();
    new_c_args.att_box = gromet.modules[0].attributes[(pre_att - 1) as usize].clone();
    new_c_args.att_idx = pre_att as usize;
    create_function(
        gromet, // gromet for metadata
        nodes,  // nodes
        edges,
        meta_nodes,
        start,
        new_c_args.clone(),
    );

    // now we need to rename the contains edge for the body to body
    for edge in edges.iter_mut() {
        if edge.src == new_c_args.parent_node.node_id.clone() && edge.e_type == *"Contains" {
            edge.e_type = "Pre".to_string();
        }
    }
    *start += 1;

    // then the condition
    new_c_args.att_box = gromet.modules[0].attributes[(condition_att - 1) as usize].clone();
    new_c_args.att_idx = condition_att as usize;
    create_att_predicate(
        gromet, // gromet for metadata
        nodes,  // nodes
        edges,
        meta_nodes,
        start,
        new_c_args.clone(),
    );
    // we need to rename the contains edge to be a condition edge
    for edge in edges.iter_mut() {
        if edge.src == new_c_args.parent_node.node_id.clone() && edge.e_type == *"Contains" {
            edge.e_type = "Condition".to_string();
        }
    }
    *start += 1;

    // now we construct the body
    new_c_args.att_box = gromet.modules[0].attributes[(body_att - 1) as usize].clone();
    new_c_args.att_idx = body_att as usize;
    create_function(
        gromet, // gromet for metadata
        nodes,  // nodes
        edges,
        meta_nodes,
        start,
        new_c_args.clone(),
    );

    // now we need to rename the contains edge for the body to body
    for edge in edges.iter_mut() {
        if edge.src == new_c_args.parent_node.node_id.clone() && edge.e_type == *"Contains" {
            edge.e_type = "Body".to_string();
        }
    }
    *start += 1;

    // now we make the pil and pol ports and connect them to the conditional node
    if function_net.pil.is_some() {
        let mut port_count = 1;
        for pic in function_net.pil.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut pic_name = String::from("Pil");
            if pic.name.as_ref().is_some() {
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
                contents: att_idx,
                nbox: bf_counter,
                att_bf_idx: 0,
                box_counter: 0,
            };
            let e3 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n2.clone());
            edges.push(e3);
            if pic.metadata.as_ref().is_some() {
                metadata_idx = pic.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
        }
    }
    if function_net.pol.is_some() {
        let mut port_count = 1;
        for poc in function_net.pol.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut poc_name = String::from("Pol");
            if poc.name.as_ref().is_some() {
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
                contents: att_idx,
                nbox: bf_counter,
                att_bf_idx: 0,
                box_counter: 0,
            };
            let e5 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n3.clone());
            edges.push(e5);
            if poc.metadata.as_ref().is_some() {
                metadata_idx = poc.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
        }
    }

    // Now we start to wire these objects together there are one unique wire types and implicit wires that need to be made
    // blow is the unique one
    for wire in function_net.wfl.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_att = att_idx; // attribute index of submodule (also opi contents value)
        let src_nbox = bf_counter; // nbox value of src opi
        let src_pil_idx = src_idx;

        let tgt_idx = wire.tgt; // port index
        let tgt_pof = function_net.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
        let tgt_opo_idx = tgt_pof.id.unwrap(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
        let tgt_box = tgt_pof.r#box; // tgt sub module box number

        let tgt_att = function_net.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
            .contents
            .unwrap(); // attribute index of submodule (also opo contents value)
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
            if tgt_nbox == node.nbox as u8 {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att == node.contents as u32 {
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

    // now to perform the implicit wiring:
    for opi in cond_att_box.opi.unwrap().iter() {
        let src_id = opi.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = condition_att as usize;

        let mut cond_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            if node.n_type == "Pil" {
                // iterate through port to check for tgt
                for p in node.in_indx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (src_id as u32) == *p {
                        cond_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_id as u32) == *p {
                                cond_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if cond_src_tgt.len() == 2 {
            let e9 = Edge {
                src: cond_src_tgt[0].clone(),
                tgt: cond_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e9);
        }
    }
    for opi in body_att_box.opi.unwrap().iter() {
        let src_id = opi.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = body_att as usize;

        let mut if_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            if node.n_type == "Pil" {
                // iterate through port to check for tgt
                for p in node.in_indx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (src_id as u32) == *p {
                        if_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_id as u32) == *p {
                                if_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if if_src_tgt.len() == 2 {
            let e10 = Edge {
                src: if_src_tgt[0].clone(),
                tgt: if_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e10);
        }
    }
    // every opo in if and else statements gets mapped to a poc of the same id, every opo but the last in a predicate
    // gets mapped to a poc of the same id.

    for opo in body_att_box.opo.unwrap().iter() {
        let src_id = opo.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = body_att as usize;

        let mut if_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_id as u32) == *p {
                                if_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        for node in nodes.iter() {
            if node.n_type == "Pol" {
                // iterate through port to check for tgt
                for p in node.out_idx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (tgt_id as u32) == *p {
                        if_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        if if_src_tgt.len() == 2 {
            let e12 = Edge {
                src: if_src_tgt[0].clone(),
                tgt: if_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e12);
        }
    }
    for opi in pre_att_box.opi.unwrap().iter() {
        let src_id = opi.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = pre_att as usize;

        let mut if_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            if node.n_type == "Pil" {
                // iterate through port to check for tgt
                for p in node.in_indx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (src_id as u32) == *p {
                        if_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_id as u32) == *p {
                                if_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if if_src_tgt.len() == 2 {
            let e15 = Edge {
                src: if_src_tgt[0].clone(),
                tgt: if_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e15);
        }
    }
    // every opo in if and else statements gets mapped to a poc of the same id, every opo but the last in a predicate
    // gets mapped to a poc of the same id.

    for opo in pre_att_box.opo.unwrap().iter() {
        let src_id = opo.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = pre_att as usize;

        let mut if_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_id as u32) == *p {
                                if_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        for node in nodes.iter() {
            if node.n_type == "Pol" {
                // iterate through port to check for tgt
                for p in node.out_idx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (tgt_id as u32) == *p {
                        if_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        if if_src_tgt.len() == 2 {
            let e16 = Edge {
                src: if_src_tgt[0].clone(),
                tgt: if_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e16);
        }
    }
    // iterate through everything but last opo
    let opo_final_idx = cond_att_box.opo.clone().unwrap().len() - 1;
    for (i, opo) in cond_att_box.opo.unwrap().iter().enumerate() {
        let src_id = opo.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = condition_att as usize;

        let mut cond_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_id as u32) == *p {
                                cond_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        for node in nodes.iter() {
            if node.n_type == "Pol" {
                // iterate through port to check for tgt
                for p in node.out_idx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (tgt_id as u32) == *p {
                        cond_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        if opo_final_idx != i {
            if cond_src_tgt.len() == 2 {
                let e14 = Edge {
                    src: cond_src_tgt[0].clone(),
                    tgt: cond_src_tgt[1].clone(),
                    e_type: String::from("Wire"),
                    prop: None,
                };
                edges.push(e14);
            }
        }
    }
}

pub fn create_while_loop(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
    cond_counter: u32,
) {
    let bf_counter = c_args.bf_counter;
    let att_idx = c_args.att_idx;
    let _att_bf_idx = c_args.att_bf_idx;
    let _box_counter = c_args.box_counter;
    let function_net = c_args.att_box.clone();

    let mut metadata_idx = 0;
    let n1 = Node {
        n_type: String::from("While_Loop"),
        value: None,
        name: Some(format!("While{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: att_idx,
        nbox: bf_counter,
        att_bf_idx: 0,
        box_counter: 0,
    };
    let e1 = Edge {
        src: c_args.parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(cond_counter as usize),
    };
    nodes.push(n1.clone());
    edges.push(e1);
    if function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .metadata
        .as_ref()
        .is_some()
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
            meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
            let me1 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }

    *start += 1;

    // now we construct the condition and body of the loop, assumption: condition is predicate, and body if a function

    let condition_att = function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .condition
        .unwrap();
    let body_att = function_net.bl.as_ref().unwrap()[cond_counter as usize]
        .body
        .unwrap();

    let cond_att_box = gromet.modules[0].attributes[(condition_att - 1) as usize].clone();
    let body_att_box = gromet.modules[0].attributes[(body_att - 1) as usize].clone();

    let mut new_c_args = c_args.clone();

    // first the condition
    new_c_args.att_box = gromet.modules[0].attributes[(condition_att - 1) as usize].clone();
    new_c_args.parent_node = n1.clone();
    new_c_args.att_idx = condition_att as usize;
    create_att_predicate(
        gromet, // gromet for metadata
        nodes,  // nodes
        edges,
        meta_nodes,
        start,
        new_c_args.clone(),
    );
    // we need to rename the contains edge to be a condition edge
    for edge in edges.iter_mut() {
        if edge.src == new_c_args.parent_node.node_id.clone() && edge.e_type == *"Contains" {
            edge.e_type = "Condition".to_string();
        }
    }
    *start += 1;

    // now we construct the body
    new_c_args.att_box = gromet.modules[0].attributes[(body_att - 1) as usize].clone();
    new_c_args.att_idx = body_att as usize;
    create_function(
        gromet, // gromet for metadata
        nodes,  // nodes
        edges,
        meta_nodes,
        start,
        new_c_args.clone(),
    );

    // now we need to rename the contains edge for the body to body
    for edge in edges.iter_mut() {
        if edge.src == new_c_args.parent_node.node_id.clone() && edge.e_type == *"Contains" {
            edge.e_type = "Body".to_string();
        }
    }
    *start += 1;

    // now we make the pil and pol ports and connect them to the conditional node
    if function_net.pil.is_some() {
        let mut port_count = 1;
        for pic in function_net.pil.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut pic_name = String::from("Pil");
            if pic.name.as_ref().is_some() {
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
                contents: att_idx,
                nbox: bf_counter,
                att_bf_idx: 0,
                box_counter: 0,
            };
            let e3 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n2.clone());
            edges.push(e3);
            if pic.metadata.as_ref().is_some() {
                metadata_idx = pic.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
        }
    }
    if function_net.pol.is_some() {
        let mut port_count = 1;
        for poc in function_net.pol.as_ref().unwrap().iter() {
            // grab name if it exists
            let mut poc_name = String::from("Pol");
            if poc.name.as_ref().is_some() {
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
                contents: att_idx,
                nbox: bf_counter,
                att_bf_idx: 0,
                box_counter: 0,
            };
            let e5 = Edge {
                src: n1.node_id.clone(),
                tgt: format!("n{}", start),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            nodes.push(n3.clone());
            edges.push(e5);
            if poc.metadata.as_ref().is_some() {
                metadata_idx = poc.metadata.unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
        }
    }

    // Now we start to wire these objects together there are one unique wire types and implicit wires that need to be made
    // blow is the unique one
    for wire in function_net.wfl.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_att = att_idx; // attribute index of submodule (also opi contents value)
        let src_nbox = bf_counter; // nbox value of src opi
        let src_pil_idx = src_idx;

        let tgt_idx = wire.tgt; // port index
        let tgt_pof = function_net.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
        let tgt_opo_idx = tgt_pof.id.unwrap(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
        let tgt_box = tgt_pof.r#box; // tgt sub module box number

        let tgt_att = function_net.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
            .contents
            .unwrap(); // attribute index of submodule (also opo contents value)
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
            if tgt_nbox == node.nbox as u8 {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att == node.contents as u32 {
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

    // now to perform the implicit wiring:
    for opi in cond_att_box.opi.unwrap().iter() {
        let src_id = opi.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = condition_att as usize;

        let mut cond_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            if node.n_type == "Pil" {
                // iterate through port to check for tgt
                for p in node.in_indx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (src_id as u32) == *p {
                        cond_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_id as u32) == *p {
                                cond_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if cond_src_tgt.len() == 2 {
            let e9 = Edge {
                src: cond_src_tgt[0].clone(),
                tgt: cond_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e9);
        }
    }
    for opi in body_att_box.opi.unwrap().iter() {
        let src_id = opi.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = body_att as usize;

        let mut if_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            if node.n_type == "Pil" {
                // iterate through port to check for tgt
                for p in node.in_indx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (src_id as u32) == *p {
                        if_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (tgt_id as u32) == *p {
                                if_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if if_src_tgt.len() == 2 {
            let e10 = Edge {
                src: if_src_tgt[0].clone(),
                tgt: if_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e10);
        }
    }
    // every opo in if and else statements gets mapped to a poc of the same id, every opo but the last in a predicate
    // gets mapped to a poc of the same id.

    for opo in body_att_box.opo.unwrap().iter() {
        let src_id = opo.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = body_att as usize;

        let mut if_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_id as u32) == *p {
                                if_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        for node in nodes.iter() {
            if node.n_type == "Pol" {
                // iterate through port to check for tgt
                for p in node.out_idx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (tgt_id as u32) == *p {
                        if_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        if if_src_tgt.len() == 2 {
            let e12 = Edge {
                src: if_src_tgt[0].clone(),
                tgt: if_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: None,
            };
            edges.push(e12);
        }
    }
    // iterate through everything but last opo
    let opo_final_idx = cond_att_box.opo.clone().unwrap().len() - 1;
    for (i, opo) in cond_att_box.opo.unwrap().iter().enumerate() {
        let src_id = opo.id.unwrap();
        let tgt_id = src_id;
        let tgt_bf_counter = 0; // because of how we have defined it
        let tgt_att_idx = condition_att as usize;

        let mut cond_src_tgt: Vec<String> = vec![];

        for node in nodes.iter() {
            // make sure in correct box
            if tgt_bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att_idx == node.contents {
                    // only opo's
                    if node.n_type == "Opo" {
                        // iterate through port to check for tgt
                        for p in node.out_idx.as_ref().unwrap().iter() {
                            // push the src first, being pif
                            if (src_id as u32) == *p {
                                cond_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        for node in nodes.iter() {
            if node.n_type == "Pol" {
                // iterate through port to check for tgt
                for p in node.out_idx.as_ref().unwrap().iter() {
                    // push the src first, being pif
                    if (tgt_id as u32) == *p {
                        cond_src_tgt.push(node.node_id.clone());
                    }
                }
            }
        }
        if opo_final_idx != i {
            if cond_src_tgt.len() == 2 {
                let e14 = Edge {
                    src: cond_src_tgt[0].clone(),
                    tgt: cond_src_tgt[1].clone(),
                    e_type: String::from("Wire"),
                    prop: None,
                };
                edges.push(e14);
            }
        }
    }
}

// This needs to be updated to handle the new node structure and remove the overloaded contents field which will mess with the wiring alot
#[allow(unused_assignments)]
pub fn create_att_expression(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    let mut metadata_idx = 0;

    let n1 = Node {
        n_type: String::from("Expression"),
        value: None,
        name: Some(format!("Expression{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: c_args.att_idx, // this is the attribute index, USED TO BE PARENT ATTRIBUTE INDEX
        nbox: c_args.bf_counter,  // inherited from parent
        att_bf_idx: c_args.att_bf_idx, // This is a reference to the index in parent function
        box_counter: c_args.box_counter,
    };
    let e1 = Edge {
        src: c_args.parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(c_args.att_idx),
    };
    nodes.push(n1.clone());
    edges.push(e1);

    let exp_box = c_args.cur_box.clone();

    if exp_box.metadata.as_ref().is_some() {
        metadata_idx = exp_box.metadata.unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
    *start += 1;

    let att_box = gromet.modules[0].attributes[c_args.att_idx - 1].clone(); // current expression attribute
    let mut parent_att_box = att_box.clone();
    if c_args.att_bf_idx != 0 {
        parent_att_box = gromet.modules[0].attributes[c_args.att_bf_idx - 1].clone();
    // parent attribute
    } else {
        parent_att_box = gromet.modules[0].r#fn.clone();
    }

    // construct opo nodes, if not none
    // not calling the opo port constuctors since they are based on grabbing in the name from top level of the gromet,
    // not a parental attribute
    if att_box.opo.is_some() {
        // grab name which is one level up and based on indexing
        // this can be done by the parent nodes contents field should give the index of the attributes
        // this constructs a vec for the names in the pofs, if any.
        let mut opo_name: Vec<String> = vec![];
        if c_args.att_bf_idx != 0 {
            for port in parent_att_box.pof.as_ref().unwrap().iter() {
                if port.r#box == c_args.box_counter as u8 {
                    if port.name.is_some() {
                        opo_name.push(port.name.as_ref().unwrap().clone());
                    } else {
                        opo_name.push(String::from("un-named"));
                    }
                }
            }
        } else if gromet.modules[0].r#fn.pof.is_some() {
            for port in gromet.modules[0].r#fn.pof.as_ref().unwrap().iter() {
                if port.r#box == c_args.bf_counter as u8 {
                    if port.name.is_some() {
                        opo_name.push(port.name.as_ref().unwrap().clone());
                    } else {
                        opo_name.push(String::from("un-named"));
                    }
                }
            }
        }
        if !opo_name.clone().is_empty() {
            let mut oport: u32 = 0;
            for _op in att_box.opo.as_ref().unwrap().iter() {
                let n2 = Node {
                    n_type: String::from("Opo"),
                    value: None,
                    name: Some(opo_name[oport as usize].clone()),
                    node_id: format!("n{}", start),
                    out_idx: Some([oport + 1].to_vec()),
                    in_indx: None,
                    contents: c_args.att_idx,      // current att idx
                    nbox: c_args.bf_counter,       // top parent top level idx
                    att_bf_idx: c_args.att_bf_idx, // current box idx of parent
                    box_counter: c_args.box_counter,
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
                if att_box.opo.clone().as_ref().unwrap()[oport as usize]
                    .metadata
                    .clone()
                    .as_ref()
                    .is_some()
                {
                    metadata_idx = att_box.opo.clone().unwrap()[oport as usize]
                        .metadata
                        .unwrap();
                    let mut repeat_meta = false;
                    for node in meta_nodes.iter() {
                        if node.metadata_idx == metadata_idx {
                            repeat_meta = true;
                        }
                    }
                    if !repeat_meta {
                        meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
                *start += 1;
                oport += 1;
            }
        }
    }
    // construct opi nodes, in not none
    if att_box.opi.is_some() {
        // grab name which is NOT one level up as in opo
        let mut opi_name: Vec<String> = vec![];
        for (j, port) in att_box.opi.as_ref().unwrap().iter().enumerate() {
            if port.name.is_some() {
                opi_name.push(port.name.as_ref().unwrap().clone());
            } else {
                // complicated logic to pull names from unpack pof's for these opi's
                let mut unpack_box: usize = 0;
                for (i, bf) in parent_att_box.bf.clone().unwrap().iter().enumerate() {
                    match bf.function_type {
                        FunctionType::Abstract => {
                            if bf.name.is_some()
                                && *bf.name.as_ref().unwrap() == String::from("unpack")
                            {
                                unpack_box = i + 1; // base 1 in gromet
                            }
                        }
                        _ => {}
                    }
                }
                if unpack_box != 0 {
                    let mut port_named = false;
                    for wire in parent_att_box.wff.clone().unwrap() {
                        let wire_src = wire.src as usize;
                        let wire_tgt = wire.tgt as usize;
                        let src_pif = parent_att_box.pif.clone().unwrap()[wire_src - 1].clone();
                        let tgt_pof = parent_att_box.pof.clone().unwrap()[wire_tgt - 1].clone();
                        // if pif matches this opi
                        if src_pif.r#box as usize == c_args.box_counter
                            && src_pif.id.unwrap() as usize == j + 1
                        {
                            // wire tgt is unpack
                            if tgt_pof.r#box as usize == unpack_box {
                                opi_name.push(tgt_pof.name.unwrap().clone());
                                port_named = true;
                            }
                        }
                    }
                    if !port_named {
                        opi_name.push(String::from("un-named"));
                    }
                } else {
                    opi_name.push(String::from("un-named"));
                }
            }
        }
        let mut iport: u32 = 0;
        for _op in att_box.opi.as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opi"),
                value: None,
                name: Some(opi_name[iport as usize].clone()), // I think this naming will get messed up if there are multiple ports...
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([iport + 1].to_vec()),
                contents: c_args.att_idx,
                nbox: c_args.bf_counter,
                att_bf_idx: c_args.att_bf_idx,
                box_counter: c_args.box_counter,
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
            if att_box.opi.clone().as_ref().unwrap()[iport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_some()
            {
                metadata_idx = att_box.opi.clone().unwrap()[iport as usize]
                    .metadata
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            *start += 1;
            iport += 1;
        }
    }
    // now to construct the nodes inside the expression, Literal and Primitives
    let mut new_c_args = c_args.clone();
    let mut box_counter: usize = 1;
    new_c_args.att_box = att_box.clone();
    if att_box.bf.is_some() {
        for att_sub_box in att_box.bf.as_ref().unwrap().iter() {
            new_c_args.box_counter = box_counter;
            new_c_args.cur_box = att_sub_box.clone();
            if att_sub_box.contents.is_some() {
                new_c_args.att_idx = att_sub_box.contents.unwrap() as usize;
            }
            match att_sub_box.function_type {
                FunctionType::Literal => {
                    create_att_literal(
                        gromet, // gromet for metadata
                        nodes,  // nodes
                        edges,
                        meta_nodes,
                        start,
                        new_c_args.clone(),
                    );
                }
                FunctionType::Primitive => {
                    create_att_primitive(
                        gromet, // gromet for metadata
                        nodes,  // nodes
                        edges,
                        meta_nodes,
                        start,
                        new_c_args.clone(),
                    );
                }
                _ => {}
            }
            box_counter += 1;
            *start += 1;
        }
    }
    // Now we perform the internal wiring of this branch
    internal_wiring(
        att_box.clone(),
        nodes,
        edges,
        c_args.att_idx,
        c_args.bf_counter,
    );

    // Now we also perform wopio wiring in case there is an empty expression
    if att_box.wopio.is_some() {
        wopio_wiring(att_box, nodes, edges, c_args.att_idx - 1, c_args.bf_counter);
    }
}

// This needs to be updated to handle the new node structure and remove the overloaded contents field which will mess with the wiring alot
#[allow(unused_assignments)]
pub fn create_att_predicate(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    let mut metadata_idx = 0;

    let n1 = Node {
        n_type: String::from("Predicate"),
        value: None,
        name: Some(format!("Predicate{}", start)),
        node_id: format!("n{}", start),
        out_idx: None,
        in_indx: None,
        contents: c_args.att_idx,        // this is the attribute index
        nbox: c_args.bf_counter,         // inherited from parent, should be 0
        att_bf_idx: c_args.att_bf_idx,   // inherited and should be 0
        box_counter: c_args.box_counter, // inherited and should be 0
    };
    let e1 = Edge {
        src: c_args.parent_node.node_id.clone(),
        tgt: format!("n{}", start),
        e_type: String::from("Contains"),
        prop: Some(c_args.att_idx),
    };
    nodes.push(n1.clone());
    edges.push(e1);

    let in_att_box = c_args.cur_box.clone();

    if in_att_box.metadata.as_ref().is_some() {
        metadata_idx = in_att_box.metadata.unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
    *start += 1;

    let att_box = gromet.modules[0].attributes[c_args.att_idx - 1].clone(); // current expression attribute

    // construct opo nodes, if not none
    // not calling the opo port constuctors since they are based on grabbing in the name from top level of the gromet,
    // not a parental attribute
    if att_box.opo.is_some() {
        // grab name which is one level up and based on indexing
        // this can be done by the parent nodes contents field should give the index of the attributes
        // this constructs a vec for the names in the pofs, if any.
        let mut opo_name: Vec<String> = vec![];
        for _port in att_box.opo.as_ref().unwrap().iter() {
            opo_name.push(String::from("un-named"));
        }
        let mut oport: u32 = 0;
        for _op in att_box.opo.as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opo"),
                value: None,
                name: Some(opo_name[oport as usize].clone()),
                node_id: format!("n{}", start),
                out_idx: Some([oport + 1].to_vec()),
                in_indx: None,
                contents: c_args.att_idx,      // this is the attribute index
                nbox: c_args.bf_counter,       // inherited from parent
                att_bf_idx: c_args.att_bf_idx, // This is a reference to the index in parent function
                box_counter: c_args.box_counter, // current box idx of parent
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
            if att_box.opo.clone().as_ref().unwrap()[oport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_some()
            {
                metadata_idx = att_box.opo.clone().unwrap()[oport as usize]
                    .metadata
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
            oport += 1;
        }
    }
    // construct opi nodes, in not none
    if att_box.opi.is_some() {
        // grab name which is NOT one level up as in opo
        let mut opi_name: Vec<String> = vec![];
        for _port in att_box.opi.as_ref().unwrap().iter() {
            opi_name.push(String::from("un-named"));
        }
        let mut iport: u32 = 0;
        for _op in att_box.opi.as_ref().unwrap().iter() {
            let n2 = Node {
                n_type: String::from("Opi"),
                value: None,
                name: Some(opi_name[iport as usize].clone()), // I think this naming will get messed up if there are multiple ports...
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([iport + 1].to_vec()),
                contents: c_args.att_idx,      // this is the attribute index
                nbox: c_args.bf_counter,       // inherited from parent
                att_bf_idx: c_args.att_bf_idx, // This is a reference to the index in parent function
                box_counter: c_args.box_counter,
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
            if att_box.opi.clone().as_ref().unwrap()[iport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_some()
            {
                metadata_idx = att_box.opi.clone().unwrap()[iport as usize]
                    .metadata
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
                    let me1 = Edge {
                        src: n2.node_id.clone(),
                        tgt: format!("m{}", metadata_idx),
                        e_type: String::from("Metadata"),
                        prop: None,
                    };
                    edges.push(me1);
                }
            }
            *start += 1;
            iport += 1;
        }
    }
    // now to construct the nodes inside the expression, Literal and Primitives
    let mut new_c_args = c_args.clone();
    let mut box_counter: usize = 1;
    new_c_args.att_box = att_box.clone();
    if att_box.bf.as_ref().is_some() {
        for sub_att_box in att_box.bf.as_ref().unwrap().iter() {
            new_c_args.box_counter = box_counter;
            new_c_args.cur_box = sub_att_box.clone();
            match sub_att_box.function_type {
                FunctionType::Literal => {
                    create_att_literal(
                        gromet, // gromet for metadata
                        nodes,  // nodes
                        edges,
                        meta_nodes,
                        start,
                        new_c_args.clone(),
                    );
                }
                FunctionType::Primitive => {
                    create_att_primitive(
                        gromet, // gromet for metadata
                        nodes,  // nodes
                        edges,
                        meta_nodes,
                        start,
                        new_c_args.clone(),
                    );
                }
                _ => {}
            }
            box_counter += 1;
            *start += 1;
        }
    }
    // Now we perform the internal wiring of this branch
    internal_wiring(
        att_box,
        &mut nodes.clone(),
        edges,
        c_args.att_idx,
        c_args.bf_counter,
    );
}
#[allow(unused_assignments)]
pub fn create_att_literal(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    let att_box = c_args.att_box;
    let lit_box = c_args.cur_box;
    // first find the pof's for box
    let mut pof: Vec<u32> = vec![];
    if att_box.pof.is_some() {
        let mut po_idx: u32 = 1;
        for port in att_box.pof.unwrap().iter() {
            if c_args.box_counter != 0 {
                if port.r#box == c_args.box_counter as u8 {
                    pof.push(po_idx);
                }
                po_idx += 1;
            } else {
                if port.r#box == c_args.bf_counter as u8 {
                    pof.push(po_idx);
                }
                po_idx += 1;
            }
        }
    }
    // now make the node with the port information
    let mut metadata_idx = 0;
    let n3 = Node {
        n_type: String::from("Literal"),
        value: Some(lit_box.value.clone().unwrap()),
        name: None,
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: None, // literals should only have out ports
        contents: c_args.att_idx,
        nbox: c_args.bf_counter,
        att_bf_idx: c_args.att_bf_idx,
        box_counter: c_args.box_counter,
    };
    nodes.push(n3.clone());
    // make edge connecting to expression
    let e4 = Edge {
        src: c_args.parent_node.node_id,
        tgt: n3.node_id.clone(),
        e_type: String::from("Contains"),
        prop: None,
    };
    edges.push(e4);
    if lit_box.metadata.is_some() {
        metadata_idx = lit_box.metadata.unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
            let me1 = Edge {
                src: n3.node_id,
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }
    *start += 1;
}

pub fn create_att_primitive(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    // first find the pof's for box
    let mut pof: Vec<u32> = vec![];
    if c_args.att_box.pof.is_some() {
        let mut po_idx: u32 = 1;
        for port in c_args.att_box.pof.clone().unwrap().iter() {
            if port.r#box == c_args.box_counter as u8 {
                pof.push(po_idx);
            }
            po_idx += 1;
        }
    }
    // then find pif's for box
    let mut pif: Vec<u32> = vec![];
    if c_args.att_box.pif.is_some() {
        let mut pi_idx: u32 = 1;
        for port in c_args.att_box.pif.unwrap().iter() {
            if port.r#box == c_args.box_counter as u8 {
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
        name: c_args.cur_box.name.clone(),
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: Some(pif),
        contents: c_args.att_idx,
        nbox: c_args.bf_counter,
        att_bf_idx: c_args.att_bf_idx,
        box_counter: c_args.box_counter,
    };
    nodes.push(n3.clone());
    // make edge connecting to expression
    let e4 = Edge {
        src: c_args.parent_node.node_id,
        tgt: n3.node_id.clone(),
        e_type: String::from("Contains"),
        prop: None,
    };
    edges.push(e4);
    if c_args.cur_box.metadata.is_some() {
        metadata_idx = c_args.cur_box.metadata.unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
            let me1 = Edge {
                src: n3.node_id,
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }
    *start += 1;
}

// This constructs abtract nodes (Primarily packs and unpacks)
// I continue to treat these as a Primitive nodes to simplify the wiring.
pub fn create_att_abstract(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    // first find the pof's for box
    let mut pof: Vec<u32> = vec![];
    if c_args.att_box.pof.is_some() {
        let mut po_idx: u32 = 1;
        for port in c_args.att_box.pof.clone().unwrap().iter() {
            if port.r#box == c_args.box_counter as u8 {
                pof.push(po_idx);
            }
            po_idx += 1;
        }
    }
    // then find pif's for box
    let mut pif: Vec<u32> = vec![];
    if c_args.att_box.pif.is_some() {
        let mut pi_idx: u32 = 1;
        for port in c_args.att_box.pif.unwrap().iter() {
            if port.r#box == c_args.box_counter as u8 {
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
        name: c_args.cur_box.name.clone(),
        node_id: format!("n{}", start),
        out_idx: Some(pof),
        in_indx: Some(pif),
        contents: c_args.att_idx,
        nbox: c_args.bf_counter,
        att_bf_idx: c_args.att_bf_idx,
        box_counter: c_args.box_counter,
    };
    nodes.push(n3.clone());
    // make edge connecting to expression
    let e4 = Edge {
        src: c_args.parent_node.node_id,
        tgt: n3.node_id.clone(),
        e_type: String::from("Contains"),
        prop: None,
    };
    edges.push(e4);
    if c_args.cur_box.metadata.is_some() {
        metadata_idx = c_args.cur_box.metadata.unwrap();
        let mut repeat_meta = false;
        for node in meta_nodes.iter() {
            if node.metadata_idx == metadata_idx {
                repeat_meta = true;
            }
        }
        if !repeat_meta {
            meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
            let me1 = Edge {
                src: n3.node_id,
                tgt: format!("m{}", metadata_idx),
                e_type: String::from("Metadata"),
                prop: None,
            };
            edges.push(me1);
        }
    }
    *start += 1;
}

// This is for the construction of Opo's
#[allow(unused_assignments)]
pub fn create_opo(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    let att_box = gromet.modules[0].attributes[c_args.parent_node.contents - 1].clone();
    // construct opo nodes, if not none
    if att_box.opo.is_some() {
        // grab name which is one level up and based on indexing
        let mut opo_name = "un-named";
        let mut oport: u32 = 0;
        for op in att_box.opo.as_ref().unwrap().iter() {
            if op.name.as_ref().is_none() && gromet.modules[0].r#fn.pof.as_ref().is_some() {
                for port in gromet.modules[0].r#fn.pof.as_ref().unwrap().iter() {
                    if port.r#box == c_args.bf_counter as u8
                        && oport == (port.id.unwrap() as u32 - 1)
                        && port.name.is_some()
                    {
                        opo_name = port.name.as_ref().unwrap();
                    }
                }
            } else if op.name.as_ref().is_some() {
                opo_name = op.name.as_ref().unwrap();
            }
            let n2 = Node {
                n_type: String::from("Opo"),
                value: None,
                name: Some(String::from(opo_name)),
                node_id: format!("n{}", start),
                out_idx: Some([oport + 1].to_vec()),
                in_indx: None,
                contents: c_args.parent_node.contents,
                nbox: c_args.bf_counter,
                att_bf_idx: c_args.parent_node.att_bf_idx,
                box_counter: c_args.parent_node.box_counter,
            };
            nodes.push(n2.clone());
            // construct edge: expression -> Opo
            let e3 = Edge {
                src: c_args.parent_node.node_id.clone(),
                tgt: n2.node_id.clone(),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            edges.push(e3);

            let mut metadata_idx = 0;
            if att_box.opo.clone().as_ref().unwrap()[oport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_some()
            {
                metadata_idx = att_box.opo.clone().unwrap()[oport as usize]
                    .metadata
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
            oport += 1;
        }
    }
}

// This is for the construction of Opi's
#[allow(unused_assignments)]
pub fn create_opi(
    gromet: &ModuleCollection, // needed still for metadata unfortunately
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    meta_nodes: &mut Vec<MetadataNode>,
    start: &mut u32, // for node and edge indexing
    c_args: ConstructorArgs,
) {
    let att_box = gromet.modules[0].attributes[c_args.parent_node.contents - 1].clone();
    // construct opi nodes, if not none
    if att_box.opi.is_some() {
        // grab name which is one level up and based on indexing
        let mut opi_name = "un-named";
        let mut oport: u32 = 0;
        for op in att_box.opi.as_ref().unwrap().iter() {
            if op.name.as_ref().is_none() && gromet.modules[0].r#fn.pif.as_ref().is_some() {
                for port in gromet.modules[0].r#fn.pif.as_ref().unwrap().iter() {
                    if port.r#box == c_args.bf_counter as u8
                        && oport == (port.id.unwrap() as u32 - 1)
                        && port.name.is_some()
                    {
                        opi_name = port.name.as_ref().unwrap();
                    }
                }
            } else if op.name.as_ref().is_some() {
                opi_name = op.name.as_ref().unwrap();
            }
            let n2 = Node {
                n_type: String::from("Opi"),
                value: None,
                name: Some(String::from(opi_name)),
                node_id: format!("n{}", start),
                out_idx: None,
                in_indx: Some([oport + 1].to_vec()),
                contents: c_args.parent_node.contents,
                nbox: c_args.bf_counter,
                att_bf_idx: c_args.parent_node.att_bf_idx,
                box_counter: c_args.parent_node.box_counter,
            };
            nodes.push(n2.clone());
            // construct edge: expression <- Opi
            let e3 = Edge {
                src: c_args.parent_node.node_id.clone(),
                tgt: n2.node_id.clone(),
                e_type: String::from("Port_Of"),
                prop: None,
            };
            edges.push(e3);

            let mut metadata_idx = 0;
            if att_box.opi.clone().as_ref().unwrap()[oport as usize]
                .metadata
                .clone()
                .as_ref()
                .is_some()
            {
                metadata_idx = att_box.opi.clone().unwrap()[oport as usize]
                    .metadata
                    .unwrap();
                let mut repeat_meta = false;
                for node in meta_nodes.iter() {
                    if node.metadata_idx == metadata_idx {
                        repeat_meta = true;
                    }
                }
                if !repeat_meta {
                    meta_nodes.append(&mut create_metadata_node(&gromet.clone(), metadata_idx));
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
            *start += 1;
            oport += 1;
        }
    }
}
// having issues with deeply nested structure, it is breaking in the internal wiring of the function level.
// wfopi: pif -> opi
pub fn wfopi_wiring(
    eboxf: FunctionNet,
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: u32,
    bf_counter: u8,
) {
    // iterate through all wires of type
    for wire in eboxf.wfopi.unwrap().iter() {
        let mut prop = None;
        let mut wfopi_src_tgt: Vec<String> = vec![];
        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox as u8 {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx) == node.contents as u32 {
                    // only include nodes with pifs
                    if node.in_indx.is_some() {
                        // exclude opi's
                        if node.n_type != "Opi" {
                            // iterate through port to check for src
                            for (i, p) in node.in_indx.as_ref().unwrap().iter().enumerate() {
                                // push the src first, being pif
                                if (wire.src as u32) == *p {
                                    wfopi_src_tgt.push(node.node_id.clone());
                                    prop = Some(i as u32);
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
            if bf_counter == node.nbox as u8 {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx) == node.contents as u32 {
                    // only opi's
                    if node.n_type == "Opi" {
                        // iterate through port to check for tgt
                        for p in node.in_indx.as_ref().unwrap().iter() {
                            // push the tgt now, being opi
                            if (wire.tgt as u32) == *p {
                                wfopi_src_tgt.push(node.node_id.clone());
                            }
                        }
                    }
                }
            }
        }
        if wfopi_src_tgt.len() == 2 && prop.is_some() {
            let e6 = Edge {
                src: wfopi_src_tgt[0].clone(),
                tgt: wfopi_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: Some(prop.unwrap() as usize),
            };
            edges.push(e6);
        }
    }
}

pub fn wfopo_wiring(
    eboxf: FunctionNet,
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: u32,
    bf_counter: u8,
) {
    // iterate through all wires of type
    for wire in eboxf.wfopo.unwrap().iter() {
        let mut wfopo_src_tgt: Vec<String> = vec![];
        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox as u8 {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx) == node.contents as u32 {
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
            if bf_counter == node.nbox as u8 {
                // make sure only looking in current attribute nodes for srcs and tgts
                if (idx) == node.contents as u32 {
                    // only include nodes with pofs
                    if node.out_idx.is_some() {
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
}
// this is duplicating wires a ton. (01/15/23)
// shouldn't use bf_counter, should use att_bf_idx since that is variable for functions, need to pull the box from the
// ports the wires point to.
pub fn wff_wiring(
    eboxf: FunctionNet,
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: u32,       // att_idx
    bf_counter: u8, // bf_counter
) {
    // iterate through all wires of type
    for wire in eboxf.wff.unwrap().iter() {
        let mut wff_src_tgt: Vec<String> = vec![];
        let mut prop = None;

        let src_idx = wire.src; // port index

        let src_pif = eboxf.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port

        let src_box = src_pif.r#box; // src sub module box number
        let src_att = idx; // attribute index of submodule (also opi contents value)
        let src_nbox = bf_counter; // nbox value of src opi

        let tgt_idx = wire.tgt; // port index
        let tgt_pof = eboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port

        let tgt_box = tgt_pof.r#box; // tgt sub module box number
        let tgt_att = idx; // attribute index of submodule (also opo contents value)
        let tgt_nbox = bf_counter; // nbox value of tgt opo

        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if src_nbox == node.nbox as u8 {
                // make sure only looking in current attribute nodes for srcs and tgts
                if src_att == node.contents as u32 {
                    // matche the box
                    if (src_box as u32) == node.box_counter as u32 {
                        // only include nodes with pifs
                        if node.in_indx.is_some() {
                            // exclude opo's
                            if node.n_type != "Opi" {
                                // iterate through port to check for src
                                for (i, p) in node.in_indx.as_ref().unwrap().iter().enumerate() {
                                    // push the tgt
                                    if (wire.src as u32) == *p {
                                        wff_src_tgt.push(node.node_id.clone());
                                        prop = Some(i as u32);
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
            if tgt_nbox == node.nbox as u8 {
                // make sure only looking in current attribute nodes for srcs and tgts
                if tgt_att == node.contents as u32 {
                    // match internal box
                    if (tgt_box as u32) == node.box_counter as u32 {
                        // only include nodes with pofs
                        if node.out_idx.is_some() {
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
        if wff_src_tgt.len() == 2 && prop.is_some() {
            let e8 = Edge {
                src: wff_src_tgt[0].clone(),
                tgt: wff_src_tgt[1].clone(),
                e_type: String::from("Wire"),
                prop: Some(prop.unwrap() as usize),
            };
            edges.push(e8);
        }
    }
}

pub fn wopio_wiring(
    eboxf: FunctionNet,
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: usize,
    bf_counter: usize,
) {
    // iterate through all wires of type
    for wire in eboxf.wopio.unwrap().iter() {
        let mut wopio_src_tgt: Vec<String> = vec![];
        // find the src node
        for node in nodes.iter() {
            // make sure in correct box
            if bf_counter == node.nbox {
                // make sure only looking in current attribute nodes for srcs and tgts
                if idx == node.contents {
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
                if idx == node.contents {
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
}

pub fn internal_wiring(
    eboxf: FunctionNet,
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: usize,
    bf_counter: usize,
) {
    // first lets wire the wfopi, note we need to first limit ourselves
    // to only nodes in the current attribute by checking the contents field
    // and then we run find the ports that match the wire src and tgt.
    // wfopi: pif -> opi
    // wff: pif -> pof
    // wfopo: opo -> pof
    // wopio: opo -> opi

    // check if wire exists, wfopi
    if eboxf.wfopi.is_some() {
        wfopi_wiring(
            eboxf.clone(),
            &mut nodes.clone(),
            edges,
            idx as u32,
            bf_counter as u8,
        );
    }

    // check if wire exists, wfopo
    if eboxf.wfopo.is_some() {
        wfopo_wiring(
            eboxf.clone(),
            &mut nodes.clone(),
            edges,
            idx as u32,
            bf_counter as u8,
        );
    }

    // check if wire exists, wff
    if eboxf.wff.is_some() {
        wff_wiring(
            eboxf.clone(),
            &mut nodes.clone(),
            edges,
            idx as u32,
            bf_counter as u8,
        );
    }

    // check if wire exists, wopio
    if eboxf.wopio.is_some() {
        wopio_wiring(eboxf, nodes, edges, idx, bf_counter);
    }
}

// now for the wiring used for imports
// needs to handle top level and function level wiring that uses the function net at the call of the import.
pub fn import_wiring(
    gromet: &ModuleCollection,
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: usize,
    bf_counter: usize,
    parent_node: Node,
) {
    // first based on the parent_node determine if we need to grab the pof's from the top scope or a sub scope
    if parent_node.att_bf_idx == 0 && gromet.modules[0].r#fn.wff.is_some() {
        // this means top level wiring
        // iterate through all wires of type
        let nboxf = gromet.modules[0].r#fn.clone();
        for wire in nboxf.wff.unwrap().iter() {
            let mut wff_src_tgt: Vec<String> = vec![];

            let src_idx = wire.src; // port index

            let src_pif = nboxf.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port

            let src_box = src_pif.r#box; // src sub module box number
            let src_att = idx; // attribute index of submodule (also opi contents value)
            let src_nbox = bf_counter; // nbox value of src opi

            let tgt_idx = wire.tgt; // port index

            let tgt_pof = nboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port

            let tgt_box = tgt_pof.r#box; // tgt sub module box number
            let tgt_att = idx; // attribute index of submodule (also opo contents value)
            let tgt_nbox = bf_counter; // nbox value of tgt opo

            // find the src node
            for node in nodes.iter() {
                // make sure in correct box
                if src_nbox == node.nbox {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if src_att == node.contents {
                        // matche the box
                        if (src_box as u32) == node.att_bf_idx as u32 {
                            // only include nodes with pifs
                            if node.in_indx.is_some() {
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
                        if (tgt_box as u32) == node.att_bf_idx as u32 {
                            // only include nodes with pofs
                            if node.out_idx.is_some() {
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
    } else {
        // this means we are in function scope, concerns on if this is cross attributal or just internal wiring...
        let mut eboxf = gromet.modules[0].r#fn.clone();
        if parent_node.contents != 0 {
            eboxf = gromet.modules[0].attributes[parent_node.contents - 1].clone();
        }

        // iterate through all wires of type
        if eboxf.wff.is_some() {
            for wire in eboxf.wff.unwrap().iter() {
                let mut wff_src_tgt: Vec<String> = vec![];

                let src_idx = wire.src; // port index

                let src_pif = eboxf.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port

                let src_box = src_pif.r#box; // src sub module box number
                let src_att = idx; // attribute index of submodule (also opi contents value)
                let src_nbox = bf_counter; // nbox value of src opi

                let tgt_idx = wire.tgt; // port index

                let tgt_pof = eboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port

                let tgt_box = tgt_pof.r#box; // tgt sub module box number
                let tgt_att = idx; // attribute index of submodule (also opo contents value)
                let tgt_nbox = bf_counter; // nbox value of tgt opo

                // find the src node
                for node in nodes.iter() {
                    // make sure in correct box
                    if src_nbox == node.nbox {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if src_att == node.contents {
                            // matche the box
                            if (src_box as u32) == node.att_bf_idx as u32 {
                                // only include nodes with pifs
                                if node.in_indx.is_some() {
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
                            if (tgt_box as u32) == node.att_bf_idx as u32 {
                                // only include nodes with pofs
                                if node.out_idx.is_some() {
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
        }
    }
}

pub fn cross_att_wiring(
    eboxf: FunctionNet, // This is the current attribute
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: usize,        // this +1 is the current attribute index
    bf_counter: usize, // this is the current box
) {
    // wire id corresponds to subport index in list so ex: wff src.id="1" means the first opi in the list of the src.box, this is the in_idx in the opi or out_indx in opo.
    // This will have to run wfopo wfopi and wff all in order to get the cross attribual wiring that can exist in all of them, refactoring won't do much in code saving space though.
    // for cross attributal wiring they will construct the following types of wires
    // wfopi: opi -> opi
    // wff: opi -> opo
    // wfopo: opo -> opo

    // check if wire exists, wfopi
    if eboxf.wfopi.is_some() {
        wfopi_cross_att_wiring(
            eboxf.clone(),
            &mut nodes.clone(),
            edges,
            idx as u32,
            bf_counter as u8,
        );
    }

    // check if wire exists, wfopo
    if eboxf.wfopo.is_some() {
        wfopo_cross_att_wiring(
            eboxf.clone(),
            &mut nodes.clone(),
            edges,
            idx as u32,
            bf_counter as u8,
        );
    }

    // check if wire exists, wff
    if eboxf.wff.is_some() {
        wff_cross_att_wiring(
            eboxf,
            &mut nodes.clone(),
            edges,
            idx as u32,
            bf_counter as u8,
        );
    }
}
// this will construct connections from the function opi's to the sub module opi's, tracing inputs through the function
// opi(sub)->opi(fun)
pub fn wfopi_cross_att_wiring(
    eboxf: FunctionNet, // This is the current attribute
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: u32,       // this is the current attribute index
    bf_counter: u8, // this is the current box
) {
    for wire in eboxf.wfopi.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_pif = eboxf.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port
        let src_opi_idx = src_pif.id.unwrap(); // index of opi port in opi list in src sub module (also opi node in_indx value)
        let src_box = src_pif.r#box; // src sub module box number, should also be the att_bf_idx of the opi

        // make sure it's a cross attributal wire and not internal
        if eboxf.bf.as_ref().unwrap()[(src_box - 1) as usize]
            .contents
            .clone()
            .is_some()
        {
            let src_att = eboxf.bf.as_ref().unwrap()[(src_box - 1) as usize]
                .contents
                .unwrap(); // attribute index of submodule (also opi contents value)
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
                if src_nbox == node.nbox as u8 {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if src_att == node.contents as u32 {
                        // make sure box index also lines up
                        if (src_box as u32) == node.box_counter as u32 {
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
                if tgt_nbox == node.nbox as u8 {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att == node.contents as u32 {
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
}
// this will construct connections from the function opo's to the sub module opo's, tracing outputs through the function
// opo(fun)->opo(sub)
pub fn wfopo_cross_att_wiring(
    eboxf: FunctionNet, // This is the current attribute
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: u32,       // this +1 is the current attribute index
    bf_counter: u8, // this is the current box
) {
    for wire in eboxf.wfopo.as_ref().unwrap().iter() {
        // collect info to identify the opo tgt node
        let tgt_idx = wire.tgt; // port index
        let tgt_pof = eboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
        let tgt_opo_idx = tgt_pof.id.unwrap(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
        let tgt_box = tgt_pof.r#box; // tgt sub module box number

        // make sure its a cross attributal wiring and not an internal one
        if eboxf.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
            .contents
            .clone()
            .is_some()
        {
            let tgt_att = eboxf.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .contents
                .unwrap(); // attribute index of submodule (also opo contents value)
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
                if src_nbox == node.nbox as u8 {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if src_att == node.contents as u32 {
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
                if tgt_nbox == node.nbox as u8 {
                    // make sure only looking in current attribute nodes for srcs and tgts
                    if tgt_att == node.contents as u32
                        && (tgt_box as u32) == node.box_counter as u32
                    {
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
}
// this will construct connections from the sub function modules opi's to another sub module opo's, tracing data inside the function
// opi(sub)->opo(sub)
#[allow(unused_assignments)]
pub fn wff_cross_att_wiring(
    eboxf: FunctionNet, // This is the current attribute, should be the function if in a function
    nodes: &mut Vec<Node>,
    edges: &mut Vec<Edge>,
    idx: u32,       // this +1 is the current attribute index
    bf_counter: u8, // this is the current box
) {
    for wire in eboxf.wff.as_ref().unwrap().iter() {
        // collect info to identify the opi src node
        let src_idx = wire.src; // port index
        let src_pif = eboxf.pif.as_ref().unwrap()[(src_idx - 1) as usize].clone(); // src port
        let src_opi_idx = src_pif.id.unwrap(); // index of opi port in opi list in src sub module (also opi node in_indx value)
        let src_box = src_pif.r#box; // src sub module box number
                                     // make sure its a cross attributal wiring and not an internal wire
        let mut src_att = idx;
        // first conditional for if we are wiring from expression or function
        if eboxf.bf.as_ref().unwrap()[(src_box - 1) as usize]
            .contents
            .clone()
            .is_some()
        {
            src_att = eboxf.bf.as_ref().unwrap()[(src_box - 1) as usize]
                .contents
                .unwrap(); // attribute index of submodule (also opi contents value)
            let src_nbox = bf_counter; // nbox value of src opi
                                       // collect info to identify the opo tgt node
            let tgt_idx = wire.tgt; // port index
            let tgt_pof = eboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
            let tgt_opo_idx = tgt_pof.id.unwrap(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
            let tgt_box = tgt_pof.r#box; // tgt sub module box number
                                         // make sure its a cross attributal wiring and not an internal wire
                                         // initialize the tgt_att for case of opo or primitive/literal source
            let mut tgt_att = idx;
            // next expression for if we are wiring to an expression or function
            if eboxf.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .contents
                .clone()
                .is_some()
            {
                tgt_att = eboxf.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                    .contents
                    .unwrap(); // attribute index of submodule (also opo contents value)
                let tgt_nbox = bf_counter; // nbox value of tgt opo
                                           // now to construct the wire
                let mut wff_src_tgt: Vec<String> = vec![];
                // find the src node
                // perhaps add a conditional on if the tgt and src att's are the same no wiring is done?
                // making sure the wiring is cross attributal and therefore won't get double wired from
                // internal wiring module as well?
                for node in nodes.iter() {
                    // make sure in correct box
                    if src_nbox == node.nbox as u8 {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if src_att == node.contents as u32
                            && (src_box as u32) == node.box_counter as u32
                        {
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
                    if tgt_nbox == node.nbox as u8 {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if tgt_att == node.contents as u32
                            && (tgt_box as u32) == node.box_counter as u32
                        {
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
            } else {
                let tgt_nbox = bf_counter; // nbox value of tgt opo
                                           // now to construct the wire
                let mut wff_src_tgt: Vec<String> = vec![];
                // find the src node
                for node in nodes.iter() {
                    // make sure in correct box
                    if src_nbox == node.nbox as u8 {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if src_att == node.contents as u32
                            && (src_box as u32) == node.box_counter as u32
                        {
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
                    if tgt_nbox == node.nbox as u8 {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if tgt_att == node.contents as u32
                            && (tgt_box as u32) == node.box_counter as u32
                        {
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
                if wff_src_tgt.len() == 2 {
                    // now we perform a conditional for naming the opi's based on
                    // the primitives pof names, we have the primitive, the opi node id
                    // and the tgt_opo_idx which is the pof idx for the name
                    for i in 0..nodes.clone().len() {
                        if nodes[i].node_id.clone() == wff_src_tgt[0].clone()
                            && eboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize]
                                .name
                                .is_some()
                        {
                            nodes[i].name = eboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize]
                                .name
                                .clone();
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
            let tgt_pof = eboxf.pof.as_ref().unwrap()[(tgt_idx - 1) as usize].clone(); // tgt port
            let tgt_opo_idx = tgt_pof.id.unwrap(); // index of tgt port in opo list in tgt sub module (also opo node out_idx value)
            let tgt_box = tgt_pof.r#box; // tgt sub module box number
                                         // make sure its a cross attributal wiring and not an internal wire
                                         // initialize the tgt_att for case of opo or primitive/literal source
            let mut tgt_att = idx;
            if eboxf.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                .contents
                .clone()
                .is_some()
            {
                tgt_att = eboxf.bf.as_ref().unwrap()[(tgt_box - 1) as usize]
                    .contents
                    .unwrap(); // attribute index of submodule (also opo contents value)
                let tgt_nbox = bf_counter; // nbox value of tgt opo
                                           // now to construct the wire
                let mut wff_src_tgt: Vec<String> = vec![];
                // find the src node
                for node in nodes.iter() {
                    // make sure in correct box
                    if src_nbox == node.nbox as u8 {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if src_att == node.contents as u32
                            && (src_box as u32) == node.box_counter as u32
                        {
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
                for node in nodes.iter() {
                    // make sure in correct box
                    if tgt_nbox == node.nbox as u8 {
                        // make sure only looking in current attribute nodes for srcs and tgts
                        if tgt_att == node.contents as u32
                            && (tgt_box as u32) == node.box_counter as u32
                        {
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
}
// external wiring is the wiring between boxes at the module level
pub fn external_wiring(gromet: &ModuleCollection, nodes: &mut Vec<Node>, edges: &mut Vec<Edge>) {
    if gromet.modules[0].r#fn.wff.as_ref().is_some() {
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
                    .unwrap();
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
                    .unwrap();
            }
            let mut wff_src_tgt: Vec<String> = vec![];
            // This is double counting since only check is name and box, check on attributes?
            // find the src
            for node in nodes.iter() {
                if node.nbox == src_box as usize
                    && src_att == node.contents as u32
                    && (node.n_type == "Opi" || node.n_type == "Import")
                {
                    for p in node.in_indx.as_ref().unwrap().iter() {
                        // push the src
                        if (src_id as u32) == *p {
                            wff_src_tgt.push(node.node_id.clone());
                        }
                    }
                }
            }
            // find the tgt
            let mut tgt_found = false; // for expression wiring the literal and opo's can be double counted sometimes
            for node in nodes.iter() {
                // check this field
                if node.n_type == "Opo" {
                    if node.nbox == tgt_box as usize {
                        if tgt_att == node.contents as u32 {
                            for p in node.out_idx.as_ref().unwrap().iter() {
                                // push the tgt
                                if (tgt_id as u32) == *p {
                                    wff_src_tgt.push(node.node_id.clone());
                                    tgt_found = true;
                                }
                            }
                        } else {
                            // perform extra check to make sure not getting sub cross attributal nodes
                            if tgt_att == node.contents as u32 {
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
                } else if (node.n_type == "Literal"
                    || node.n_type == "Import"
                    || node.n_type == "Primitive")
                    && !tgt_found
                    && node.nbox == tgt_box as usize
                {
                    for p in node.out_idx.as_ref().unwrap().iter() {
                        // push the tgt
                        if (tgt_id as u32) == *p {
                            wff_src_tgt.push(node.node_id.clone());
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
}

pub fn parse_gromet_queries(gromet: ModuleCollection) -> Vec<String> {
    let mut queries: Vec<String> = vec![];

    let start: u32 = 0;

    queries.append(&mut create_module(&gromet));
    queries.append(&mut create_graph_queries(&gromet, start));

    queries
}
