//! Program to parse MathML

use clap::Parser;
use mathml::ast::Math;
use petgraph::dot::{Config, Dot};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input file containing MathML
    input: String,

    /// Whether to normalize the output (collapse redundant mrows, collapse subscripts, etc.)
    #[arg(long, default_value_t = false)]
    normalize: bool,
}

fn main() {
    let args = Cli::parse();
    let input = &args.input;
    let contents = std::fs::read_to_string(input)
        .unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let mut math = contents.parse::<Math>().unwrap();
    //parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    if args.normalize {
        math.normalize();
    }
    let g = math.to_graph();
    println!("{}", Dot::with_config(&g, &[Config::EdgeNoLabel]));
}
