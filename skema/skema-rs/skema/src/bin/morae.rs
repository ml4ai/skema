use clap::Parser;
use mathml::mml2pn::get_mathml_asts_from_file;
pub use mathml::mml2pn::{ACSet, Term};

#[cfg(test)]
use std::fs;

// new imports
use mathml::acset::{PetriNet, RegNet};

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

    if new_args.arg == *"auto" {
        if new_args.model_id.is_some() {
            module_id = new_args.model_id.unwrap();
        }

        let host = "localhost";

        let math_content = module_id2mathml_ast(module_id, host);

        // This does get a panic with a message, so need to figure out how to forward it
        //let mathml_ast =
        get_mathml_asts_from_file("../../data/mml2pn_inputs/ta4_sir_v1/mml_list.txt");

        println!("\nmath_content: {:?}", math_content);
        //println!("\nmathml_ast: {:?}", mathml_ast);

        println!("\nPN from code: {:?}", ACSet::from(math_content.clone()));
        //println!("\nPN from mathml: {:?}\n", ACSet::from(mathml_ast.clone()));

        println!(
            "\nAMR from code: {:?}",
            PetriNet::from(ACSet::from(math_content))
        );
        /*println!(
            "\nAMR from mathml: {:?}\n",
            PetriNet::from(ACSet::from(mathml_ast))
        );*/
    }
    // This is the graph id for the top level function for the core dynamics for our test case.
    else if new_args.arg == *"manual" {
        // still need to grab the module id

        let host = "localhost";

        let (core_dynamics_ast, _metadata_map_ast) =
            subgraph2_core_dyn_ast(module_id, host).unwrap();

        // expressions that are working: 5 (T), 6 (H)
        // 7 (E) should be fixable, the USub is not being subsituted and is at the end of the RHS instead of the start

        println!("\n Ast:\n {:?}", core_dynamics_ast[5].clone());
    } else if new_args.arg == *"AMR_test" {
        let mathml_asts =
            get_mathml_asts_from_file("../../data/mml2pn_inputs/lotka_voltera/mml_list.txt");
        let regnet = RegNet::from(mathml_asts);
        println!("\nRegnet AMR: {:?}\n", regnet.clone());
        let regnet_serial = serde_json::to_string(&regnet).unwrap();
        println!("For serialization test:\n\n {}", regnet_serial);

        let mathml_pn_asts =
            get_mathml_asts_from_file("../../data/mml2pn_inputs/simple_sir_v1/mml_list.txt");
        let pn = PetriNet::from(ACSet::from(mathml_pn_asts));
        println!("\nPetriNet AMR: {:?}", pn);
        let pn_serial = serde_json::to_string(&pn).unwrap();
        println!("For serialization test:\n\n {}", pn_serial);
    } else {
        println!("Unknown Command!");
    }
}

#[test]
fn test_lotka_volterra_RegNet() {
    let mathml_asts =
        get_mathml_asts_from_file("../../../data/mml2pn_inputs/lotka_volterra/mml_list.txt");
    let regnet = RegNet::from(mathml_asts);

    let file_contents =
        fs::read_to_string("../../../skema/skema-rs/mathml/tests/lotka_volterra_regnet.json")
            .expect("Unable to read file");
    let regnet_output: RegNet = serde_json::from_str(&file_contents).unwrap();

    assert_eq!(regnet_output, regnet);
}
