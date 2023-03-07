//! Program to get comments from a source code file.
//! Original logic for handling Fortran code was implemented by Saumya Debray in Python for the
//! AutoMATES project (https://ml4ai.github.io/automates), and ported over to this Rust version by
//! Adarsh Pyarelal for the SKEMA project.

use clap::Parser;
use std::fs::write;
use std::path::Path;

use comment_extraction::conventions::dssat::get_comments as get_fortran_comments;
use comment_extraction::languages::python::get_comments as get_python_comments;
use comment_extraction::languages::cpp::get_comments as
get_cpp_comments;

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input source code file
    input: String,

    /// Optional path to output file. If this is not specified, the program will print the
    /// output to the standard output instead of writing to a file.
    output: Option<String>,
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

        if let Some(path) = args.output {
            write(path, comments).expect("Unable to write to file!");
        } else {
            println!("{:?}", comments);
        }
    } else if extension == "py" {
        let comments = get_python_comments(input);
        let comments = serde_json::to_string(&comments).unwrap();
        if let Some(path) = args.output {
            write(path, comments).expect("Unable to write to file!");
        } else {
            println!("{:?}", comments);
        }
    } else if extension == "cpp" || extension == "c" {
        let comments = get_cpp_comments(input);
        let comments = serde_json::to_string(&comments).unwrap();
        if let Some(path) = args.output {
            write(path, comments).expect("Unable to write to file!");
        } else {
            println!("{:?}", comments);
        }
    } else {
        panic!(
            "Unable to infer programming language for file \"{input}\"! \
            File extension must be one of the following: {{\".py\", \".f\", \".for\"}}"
        )
    }
}
