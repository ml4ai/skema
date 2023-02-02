///! Program to parse MathML and convert it to a Petri Net
use clap::Parser;
use mathml::{
    ast::Math,
    mml2pn::{export_eqn_dict_json, mathml_asts_to_eqn_dict},
    parsing::parse,
};
use std::{
    fs::File,
    io::{self, BufRead},
};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input file containing MathML
    input: String,

    /// Whether to normalize the output (collapse redundant mrows, collapse subscripts, etc.)
    #[arg(long, default_value_t = false)]
    normalize: bool,
}

fn process_file(filepath: &str) {
    let f = File::open(filepath).unwrap();
    let lines = io::BufReader::new(f).lines();

    let mut mathml_asts = Vec::<Math>::new();

    for line in lines {
        if let Ok(l) = line {
            if let Some('#') = &l.chars().nth(0) {
                // Ignore lines starting with '#'
            } else {
                // Parse MathML into AST
                let (_, math) = parse(&l).unwrap_or_else(|_| panic!("Unable to parse line {}!", l));
                mathml_asts.push(math);
            }
        }
    }

    let mut eqn_dict = mathml_asts_to_eqn_dict(mathml_asts);
    export_eqn_dict_json(&mut eqn_dict)
}

#[test]
fn test_simple_sir_v1() {
    process_file("../../mml2pn/mml/simple_sir_v1/mml_list.txt");
}

//#[test]
//fn test_simple_sir_v2() {
//process_file("../../mml2pn/mml/simple_sir_v2/mml_list.txt");
//}

//#[test]
//fn test_simple_sir_v3() {
//process_file("../../mml2pn/mml/simple_sir_v3/mml_list.txt");
//}

//#[test]
//fn test_simple_sir_v4() {
//process_file("../../mml2pn/mml/simple_sir_v4/mml_list.txt");
//}

fn main() {
    let args = Cli::parse();
    process_file(&args.input);
}
