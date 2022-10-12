//! Program to parse MathML

use clap::Parser;
use mathml::parsing::parse;

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input file containing MathML
    input: String,
}

fn main() {
    let args = Cli::parse();
    let input = &args.input;
    let contents = std::fs::read_to_string(input).expect("Unable to read file {input}");
    let (remaining_input, math) = parse(&contents).expect("Unable to parse file {input}!");
    dbg!(math);
}
