use clap::Parser;

use mathml::mml2pn::get_mathml_asts_from_file;
pub use mathml::mml2pn::{ACSet, Term};

// new imports
use std::env;
use skema::config::Config;
use mathml::acset::{PetriNet, RegNet};
use mathml::parsers::decapodes_serialization::{
    to_wiring_diagram, DecapodesCollection, WiringDiagram,
};
use mathml::parsers::first_order_ode::get_FirstOrderODE_vec_from_file;
use mathml::parsers::math_expression_tree::MathExpressionTree;
use skema::model_extraction::{module_id2mathml_MET_ast, subgraph2_core_dyn_ast};
use std::fs::File;
use std::io::{BufRead, BufReader};
use neo4rs::{query, Graph, Node, UnboundedRelation, Relation};
use tokio::task;
use tokio::runtime::Builder;
use std::sync::Arc;

#[derive(Parser, Debug)]
struct Cli {
    /// the commandline arg
    #[arg(short, long, default_value_t = String::from("auto"))]
    arg: String,

    #[arg(short, long)]
    model_id: Option<i64>,
}

pub async fn module_query(config: Config) -> Vec<i64> {
    // Connect to Memgraph.

    let graph = Arc::new(config.graphdb_connection().await);

    let mut ids = Vec::<i64>::new();
    let mut result = graph.execute(
        query("MATCH (n:Module) RETURN n")).await.unwrap();
    while let Ok(Some(row)) = result.next().await {
        let node: Node = row.get("n").unwrap();
        ids.push(node.id());
    }

    ids
}

#[tokio::main]
async fn main() {
    // setup command line argument on if the core dynamics has been found manually or needs to be found automatically
    /*
    Command line args;
        - auto -> This will attempt an automated search for the dynamics
        - manual -> This assumes the input is the function of the dynamics
    */
    let new_args = Cli::parse();

    //let mut module_id = 883;
    let mut module_id = 2431;
    // now to prototype an algorithm to find the function that contains the core dynamics

    if new_args.arg == *"auto" {
        if new_args.model_id.is_some() {
            module_id = new_args.model_id.unwrap();
        }

        /*let db_protocol = env::var("SKEMA_GRAPH_DB_PROTO").unwrap_or("bolt+s://".to_string());
        let db_host = env::var("SKEMA_GRAPH_DB_HOST").unwrap_or("graphdb-bolt.askem.lum.ai".to_string());
        let db_port = env::var("SKEMA_GRAPH_DB_PORT").unwrap_or("443".to_string());*/
    
        let db_protocol = env::var("SKEMA_GRAPH_DB_PROTO").unwrap_or("bolt+s://".to_string());
        let db_host = env::var("SKEMA_GRAPH_DB_HOST").unwrap_or("graphdb-bolt.askem.lum.ai".to_string());
        let db_port = env::var("SKEMA_GRAPH_DB_PORT").unwrap_or("443".to_string());

        let config = Config {
            db_protocol: db_protocol.clone(),
            db_host: db_host.clone(),
            db_port: db_port.parse::<u16>().unwrap(),
        };

        let response = module_query(config.clone()).await;
        println!("{:?}", response.clone());

        let mut ids = Vec::<i64>::new();
        let graph = Arc::new(config.graphdb_connection().await);
        let mut result = graph.execute(
            query("MATCH (n:Module) RETURN n")).await.unwrap();
        println!("got here");
        while let Ok(Some(row)) = result.next().await {
            let node: Node = row.get("n").unwrap();
            ids.push(node.id());
            println!("{:?}", node.id());
        }
        println!("{:?}", ids.clone());
        let math_content = module_id2mathml_MET_ast(ids[1], config.clone()).await;
        println!("{:?}", math_content.clone());
        println!("\nAMR from code: {:?}", PetriNet::from(math_content));

        //let input_src = "../../data/mml2pn_inputs/testing_eqns/sidarthe_mml.txt";

        // This does get a panic with a message, so need to figure out how to forward it
        //let _mathml_ast = get_mathml_asts_from_file(input_src.clone());

        /*let f = File::open(input_src.clone()).unwrap();
        let lines = BufReader::new(f).lines();
        let mut deca_vec = Vec::<MathExpressionTree>::new();
        let mut wiring_vec = Vec::<WiringDiagram>::new();

        for line in lines.flatten() {
            let mut deca = line
                .parse::<MathExpressionTree>()
                .unwrap_or_else(|_| panic!("Unable to parse line {}!", line));
            wiring_vec.push(to_wiring_diagram(&deca))
        }

        let decapodescollection = DecapodesCollection {
            decapodes: wiring_vec.clone()
        };

        println!("{:?}", wiring_vec.clone());
        println!("decapode collection: {:?}", decapodescollection.clone());
        */
        //let odes = get_FirstOrderODE_vec_from_file(input_src.clone());

        //println!("\nmath_content: {:?}", math_content);
        //println!("\nmathml_ast: {:?}", odes);

        //println!(
        //    "\nAMR from mathml: {}\n",
        //    serde_json::to_string(&PetriNet::from(odes)).unwrap()
        //);
        //println!("\nAMR from code: {:?}", PetriNet::from(math_content));
    }
    // This is the graph id for the top level function for the core dynamics for our test case.
}

/*#[test]
fn test_lotka_volterra_RegNet() {
    let mathml_asts =
        get_mathml_asts_from_file("../../../data/mml2pn_inputs/lotka_volterra/mml_list.txt");
    let regnet = RegNet::from(mathml_asts);

    let file_contents =
        fs::read_to_string("../../../skema/skema-rs/mathml/tests/lotka_volterra_regnet.json")
            .expect("Unable to read file");
    let regnet_output: RegNet = serde_json::from_str(&file_contents).unwrap();

    assert_eq!(regnet_output, regnet);
}*/
