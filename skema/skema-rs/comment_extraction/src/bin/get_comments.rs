// Program to get comments from a source code file.
// Original version for handling Fortran code written by Saumya Debray in Python for the AutoMATES
// project (https://ml4ai.github.io/automates)
// This Rust port was written by Adarsh Pyarelal for the SKEMA project.

use pretty_env_logger;

use clap::Parser;
use std::path::Path;

use comment_extraction::conventions::dssat::get_comments as get_fortran_comments;

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input source code file
    filepath: String,
}

fn main() {
    pretty_env_logger::init();
    let args = Cli::parse();
    let extension = Path::new(&args.filepath)
        .extension()
        .expect("Unable to get extension for {args.filepath}!")
        .to_str()
        .expect("Unable to convert extension to a valid string!");

    match get_fortran_comments(args.filepath) {
        Ok(c) => println!("{}", serde_json::to_string(&c).unwrap()),
        _ => panic!("Error getting the comments"),
    };
}
