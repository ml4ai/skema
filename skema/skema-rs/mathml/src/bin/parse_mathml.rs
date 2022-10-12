//! Program to parse MathML

use clap::Parser;
use mathml::{graph::MathMLGraph, parsing::parse};
use petgraph::dot::{Config, Dot};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input file containing MathML
    input: String,
}

fn main() {
    let args = Cli::parse();
    let input = &args.input;
    let contents = std::fs::read_to_string(input).expect("Unable to read file {input}!");
    let (remaining_input, math) = parse(&contents).expect("Unable to parse file {input}!");
    let G = math.to_graph();
    println!("{:?}", Dot::with_config(&G, &[Config::EdgeNoLabel]));
}
