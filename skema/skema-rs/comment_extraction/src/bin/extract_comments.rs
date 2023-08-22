//! Program to get comments from a source code file.

use clap::Parser;
use comment_extraction::{
    conventions::Convention,
    extraction::{extract_comments_from_directory, extract_comments_from_file},
};

/// Command line arguments
#[derive(Parser)]
struct Cli {
    /// Path to input source code file
    input: String,

    /// Convention that the codebase follows (optional).
    #[arg(long, value_enum)]
    convention: Option<Convention>,
}

fn main() {
    pretty_env_logger::init();
    let args = Cli::parse();
    let input = &args.input;
    let convention = &args.convention;

    let metadata = std::fs::metadata(input).unwrap();
    if metadata.is_file() {
        // We use unwrap below since we want the program to complain loudly and fail if only a
        // single file is given as input and the comment extraction fails.
        // In contrast, when we extract comments from a directory, we do not want the program to
        // complain for every unprocessable file.
        let comments = extract_comments_from_file(input, convention).unwrap();
        println!("{comments}");
    } else if metadata.is_dir() {
        let comments = extract_comments_from_directory(input, convention);
        println!("{comments}");
    }
}

/// Test DSSAT-style comment extraction
#[test]
fn test_dssat_style_extraction() {
    use comment_extraction::conventions;
    use serde_json as json;
    use std::fs::read_to_string;
    let output: json::Value =
        json::from_str(&read_to_string("tests/data/chime_sir_output.json").unwrap()).unwrap();
    let comments: json::Value = json::from_str(
        &json::to_string(
            &conventions::dssat::get_comments(
                "./../../../data/epidemiology/CHIME/CHIME_SIR_model/code/CHIME_SIR.for",
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(output, comments);
}