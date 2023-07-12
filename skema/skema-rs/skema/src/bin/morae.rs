use clap::Parser;
use mathml::mml2pn::get_mathml_asts_from_file;
pub use mathml::mml2pn::{ACSet, Term};
use mathml::parsers::first_order_ode::FirstOrderODE;
#[cfg(test)]
use std::fs;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use mathml::ast::operator::Operator::Multiply;
use mathml::parsers::math_expression_tree::MathExpressionTree;
use mathml::parsers::math_expression_tree::MathExpressionTree::{Atom, Cons};

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

        let input_src = "../../data/mml2pn_inputs/testing_eqns/mml_list3.txt";

        // This does get a panic with a message, so need to figure out how to forward it
        let mathml_ast = get_mathml_asts_from_file(input_src.clone());

        let odes = get_FirstOrderODE_vec_from_file(input_src.clone());

        println!("\nmath_content: {:?}", math_content);
        println!("\nmathml_ast: {:?}", mathml_ast);

        println!("\nPN from code: {:?}", ACSet::from(math_content.clone()));
        println!("\nPN from mathml: {:?}\n", RegNet::from(mathml_ast.clone()));

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

pub fn get_FirstOrderODE_vec_from_file(filepath: &str) -> Vec<FirstOrderODE> {
    let f = File::open(filepath).unwrap();
    let lines = BufReader::new(f).lines();

    let mut ode_vec = Vec::<FirstOrderODE>::new();

    for line in lines.flatten() {
        if let Some('#') = &line.chars().next() {
            // Ignore lines starting with '#'
        } else {
            // Parse MathML into FirstOrderODE
            let mut ode = line
                .parse::<FirstOrderODE>()
                .unwrap_or_else(|_| panic!("Unable to parse line {}!", line));
            println!(
                "ode_line rhs string before: {:?}\n",
                ode.rhs.to_string().clone()
            );
            //println!("ode_line rhs data struct: {:?}\n", ode.rhs.clone());
            //println!("ode_line state: {:?}\n", ode.lhs_var.to_string().clone());
            ode.rhs = flatten_mults(ode.rhs.clone());
            println!(
                "ode_line rhs string after: {:?}\n",
                ode.rhs.to_string().clone()
            );
            ode_vec.push(ode);
        }
    }
    ode_vec
}

pub fn flatten_mults(mut equation: MathExpressionTree) -> MathExpressionTree {
    match equation {
        Cons(ref x, ref mut y) => match x {
            Multiply => {
                match y[1].clone() {
                    Cons(x1, y1) => match x1 {
                        Multiply => {
                            let mut temp1 = flatten_mults(y1[0].clone());
                            let mut temp2 = flatten_mults(y1[1].clone());
                            y.remove(1);
                            y.append(&mut [temp1.clone(), temp2.clone()].to_vec())
                        }
                        _ => {
                            let mut temp1 = y1[0].clone();
                            let mut temp2 = y1[1].clone();
                            y.remove(1);
                            y.append(&mut [temp1.clone(), temp2.clone()].to_vec())
                        }
                    },
                    Atom(x1) => {}
                }
                match y[0].clone() {
                    Cons(x0, y0) => match x0 {
                        Multiply => {
                            let mut temp1 = flatten_mults(y0[0].clone());
                            let mut temp2 = flatten_mults(y0[1].clone());
                            y.remove(0);
                            y.append(&mut [temp1.clone(), temp2.clone()].to_vec());
                        }
                        _ => {
                            let mut temp1 = y0[0].clone();
                            let mut temp2 = y0[1].clone();
                            y.remove(0);
                            y.append(&mut [temp1.clone(), temp2.clone()].to_vec())
                        }
                    },
                    Atom(x0) => {}
                }
            }
            _ => {
                if y.len() > 1 {
                    let temp1 = flatten_mults(y[1].clone());
                    let temp0 = flatten_mults(y[0].clone());
                    y.remove(1);
                    y.remove(0);
                    y.append(&mut [temp0.clone(), temp1.clone()].to_vec())
                } else {
                    let temp0 = flatten_mults(y[0].clone());
                    y.remove(0);
                    y.append(&mut [temp0.clone()].to_vec())
                }
            }
        },
        Atom(ref x) => {}
    }
    equation
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
