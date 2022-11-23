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
        std::fs::read_to_string(input).expect(format!("Unable to read file {input}!").as_str());
    let (_, mut math) = parse(&contents).expect(format!("Unable to parse file {input}!").as_str());
    math.normalize();
    let g = math.to_graph();
    println!("{}", Dot::with_config(&g, &[Config::EdgeNoLabel]));
}
