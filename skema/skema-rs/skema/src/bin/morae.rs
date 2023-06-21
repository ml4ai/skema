use clap::Parser;
use mathml::ast::{Math, Operator};
use mathml::mml2pn::get_mathml_asts_from_file;
pub use mathml::mml2pn::{ACSet, Term};
use petgraph::prelude::*;
use rsmgclient::{ConnectParams, Connection, MgError, Node, Relationship, Value};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::string::ToString;
use std::{env, fs};

// new imports
use mathml::acset::{ModelRegNet, PetriNet, RegNet};
use mathml::ast::MathExpression;
use mathml::ast::MathExpression::Mo;
use mathml::ast::MathExpression::Mrow;
use mathml::petri_net::recognizers::is_add_or_subtract_operator;
use skema::model_extraction::{module_id2mathml_ast, subgraph2_core_dyn_ast};

#[derive(Parser, Debug)]
struct Cli {
    /// the commandline arg
    #[arg(short, long, default_value_t = String::from("auto"))]
    arg: String,

    #[arg(short, long)]
    model_id: Option<i64>,
}

fn main() {
    // setup command line argument on if the core dynamics has been found manually or needs to be found automatically
    /*
    Command line args;
        - auto -> This will attempt an automated search for the dynamics
        - manual -> This assumes the input is the function of the dynamics
    */
    let new_args = Cli::parse();

    let mut module_id = 2233;
    // now to prototype an algorithm to find the function that contains the core dynamics

    if new_args.arg == "auto".to_string() {
        if !new_args.model_id.is_none() {
            module_id = new_args.model_id.unwrap();
        }

        let host = "localhost";

        let math_content = module_id2mathml_ast(module_id.clone(), &host);

        let mathml_ast =
            get_mathml_asts_from_file("../../data/mml2pn_inputs/simple_sir_v1/mml_list.txt");

        println!("\nmath_content: {:?}", math_content.clone());
        println!("\nmathml_ast: {:?}", mathml_ast.clone());

        println!("\nPN from code: {:?}", ACSet::from(math_content.clone()));
        println!("\nPN from mathml: {:?}\n", ACSet::from(mathml_ast.clone()));

        println!(
            "\nAMR from code: {:?}",
            PetriNet::from(ACSet::from(math_content.clone()))
        );
        println!(
            "\nAMR from mathml: {:?}\n",
            PetriNet::from(ACSet::from(mathml_ast.clone()))
        );
    }
    // This is the graph id for the top level function for the core dynamics for our test case.
    else if new_args.arg == "manual".to_string() {
        // still need to grab the module id

        let host = "localhost";

        let (mut core_dynamics_ast, metadata_map_ast) =
            subgraph2_core_dyn_ast(module_id.clone(), &host).unwrap();

        // expressions that are working: 5 (T), 6 (H)
        // 7 (E) should be fixable, the USub is not being subsituted and is at the end of the RHS instead of the start

        println!("\n Ast:\n {:?}", core_dynamics_ast[5].clone());
    } else if new_args.arg == "AMR_test".to_string() {
        let mathml_asts =
            get_mathml_asts_from_file("../../data/mml2pn_inputs/lotka_voltera/mml_list.txt");
        let mut regnet = RegNet::from(mathml_asts);
        println!("\nRegnet AMR: {:?}\n", regnet.clone());
        regnet.model.vertices.sort();
        regnet.model.edges.sort();
        let regnet_serial = serde_json::to_string(&regnet).unwrap();
        println!("For serialization test:\n\n {}", regnet_serial);

        let mathml_pn_asts =
            get_mathml_asts_from_file("../../data/mml2pn_inputs/simple_sir_v1/mml_list.txt");
        let mut pn = PetriNet::from(ACSet::from(mathml_pn_asts));
        println!("\nPetriNet AMR: {:?}", pn.clone());
        let pn_serial = serde_json::to_string(&pn).unwrap();
        println!("For serialization test:\n\n {}", pn_serial);
    } else {
        println!("Unknown Command!");
    }
}

#[test]
fn test_lotka_voltera_RegNet() {
    let mathml_asts =
        get_mathml_asts_from_file("../../../data/mml2pn_inputs/lotka_voltera/mml_list.txt");
    let mut regnet = RegNet::from(mathml_asts);
    regnet.model.vertices.sort();
    regnet.model.edges.sort();
    let regnet_serial = serde_json::to_string(&regnet).unwrap();

    let mut file_contents =
        fs::read_to_string("../../../skema/skema-rs/mathml/tests/lotka_voltera_regnet.json")
            .expect("Unable to read file");

    assert_eq!(file_contents, regnet_serial);
}
