// Program to get comments from a source code file.
// Original version for handling Fortran code written by Saumya Debray in Python for the AutoMATES
// project (https://ml4ai.github.io/automates)
// This Rust port was written by Adarsh Pyarelal for the SKEMA project.

use pretty_env_logger;

use clap::Parser;
use std::path::Path;

use comment_extraction::conventions::dssat::get_comments as get_fortran_comments;
use comment_extraction::languages::python::get_comments as get_python_comments;

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input source code file
    filepath: String,
}

fn main() {
    pretty_env_logger::init();
    let args = Cli::parse();
    let filepath = &args.filepath;
    let extension = Path::new(filepath)
        .extension()
        .expect("Unable to get extension for {args.filepath}!")
        .to_str()
        .expect("Unable to convert extension to a valid string!");

    if extension == "f" || extension == "for" {
        let comments = get_fortran_comments(filepath).expect("Error getting the comments!");
        println!(
            "{}",
            serde_json::to_string(&comments).expect("Error serializing the comments to JSON!")
        );
    } else if extension == "py" {
        get_python_comments(filepath);
    } else {
        panic!(
            "Unable to infer programming language for file \"{filepath}\"! \
            File extension must be one of the following: {{\".py\", \".f\", \".for\"}}"
        )
    }
}
