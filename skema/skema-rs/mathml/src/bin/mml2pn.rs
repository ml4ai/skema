///! Program to parse MathML and convert it to a Petri Net
use clap::Parser;

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
    //process_file(&args.input);
}
