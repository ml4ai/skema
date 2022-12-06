//! Program to parse MathML

use clap::Parser;
use mathml::parsing::parse;
use petgraph::dot::{Config, Dot};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input file containing MathML
    input: String,
}

fn main() {
    let args = Cli::parse();
    let input = &args.input;
    let contents =
        std::fs::read_to_string(input).unwrap_or_else(|_| panic!("{}", "Unable to read file {input}!"));
    let (_, math) = parse(&contents).unwrap_or_else(|_| panic!("{}", "Unable to parse file {input}!"));
    let g = math.to_graph();
    println!("{}", Dot::with_config(&g, &[Config::EdgeNoLabel]));
}
