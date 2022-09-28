// Program to get comments from a source code file.
// Original version for handling Fortran code written by Saumya Debray in Python for the AutoMATES
// project (https://ml4ai.github.io/automates)
// This Rust port was written by Adarsh Pyarelal for the SKEMA project.

use pretty_env_logger;

use clap::Parser;
use std::path::Path;
use std::fs::write;

use comment_extraction::conventions::dssat::get_comments as get_fortran_comments;
use comment_extraction::languages::python::get_comments as get_python_comments;

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input source code file
    input: String,

    /// Path to output file
    output: String,
}

fn main() {
    pretty_env_logger::init();
    let args = Cli::parse();
    let input = &args.input;
    let extension = Path::new(input)
        .extension()
        .expect("Unable to get extension for {input}!")
        .to_str()
        .expect("Unable to convert extension to a valid string!");

    if extension == "f" || extension == "for" {
        let comments = get_fortran_comments(input).unwrap();
        let comments = serde_json::to_string(&comments).unwrap();
        write(&args.output, comments).expect("Unable to write to file!");
    } else if extension == "py" {
        let comments = get_python_comments(input);
        let comments = serde_json::to_string(&comments).unwrap();
        write(&args.output, comments).expect("Unable to write to file!");
    } else {
        panic!(
            "Unable to infer programming language for file \"{input}\"! \
            File extension must be one of the following: {{\".py\", \".f\", \".for\"}}"
        )
    }
}
