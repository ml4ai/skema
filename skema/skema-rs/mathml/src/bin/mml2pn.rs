///! Program to parse MathML and convert it to a Petri Net
use clap::{Parser, ValueEnum};
use mathml::mml2pn::ACSet;
use std::fmt;

#[derive(Debug, Clone, ValueEnum, Default)]
enum OutputFormat {
    #[default]
    Json,
    Dot,
}

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input file containing MathML
    input: String,

    /// Whether to normalize the output (collapse redundant mrows, collapse subscripts, etc.)
    #[arg(long, default_value_t = false)]
    normalize: bool,

    /// Output format
    #[clap(long, value_enum, default_value_t = OutputFormat::Json)]
    format: OutputFormat,
}

fn main() {
    let args = Cli::parse();
    let acset = ACSet::from_file(&args.input);
    match &args.format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string(&acset).unwrap());
        }
        OutputFormat::Dot => {
            println!("{}", acset.to_dot());
        }
    }
}
