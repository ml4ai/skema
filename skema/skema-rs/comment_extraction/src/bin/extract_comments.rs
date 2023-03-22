//! Program to get comments from a source code file.
//! Original logic for handling Fortran code was implemented by Saumya Debray in Python for the
//! AutoMATES project (https://ml4ai.github.io/automates), and ported over to this Rust version by
//! Adarsh Pyarelal for the SKEMA project.

use clap::Parser;
use std::path::Path;

use comment_extraction::{
    conventions::dssat::get_comments as get_fortran_comments,
    languages::{
        cpp::get_comments as get_cpp_comments, python::get_comments as get_python_comments,
    },
};

#[derive(Parser, Debug)]
struct Cli {
    /// Path to input source code file
    input: String,
}

#[derive(Debug)]
enum Language {
    /// Python
    Python,

    /// Fortran
    Fortran,

    /// C++ or C
    CppOrC,

    /// Other (unknown)
    Other,
}

/// Infer language for the file in question.
fn infer_language(filepath: &str) -> Language {
    let extension = Path::new(filepath)
        .extension()
        .expect(&format!("Unable to get extension for {}!", filepath))
        .to_str()
        .expect("Unable to convert extension to a valid string!");
    if extension == "f" || extension == "for" {
        Language::Fortran
    } else if extension == "py" {
        Language::Python
    } else if extension == "cpp" || extension == "c" {
        Language::CppOrC
    } else {
        Language::Other
    }
}

/// Extract comments from file and return them.
fn extract_comments_from_file(filepath: &str) -> Result<String, &str> {
    let language = infer_language(filepath);
    let comments: String;
    match language {
        Language::Fortran => {
            comments =
                serde_json::to_string(&get_fortran_comments(filepath).unwrap()).unwrap();
        }
        Language::Python => {
            comments = serde_json::to_string(&get_python_comments(filepath)).unwrap();
        }
        Language::CppOrC => {
            comments = serde_json::to_string(&get_cpp_comments(filepath)).unwrap();
        }
        Language::Other => {
            return Err("File extension does not correspond to any of the ones the comment extractor handles:{'py', 'f', 'for', 'c', 'cpp'}")
        }
    }

    Ok(comments)
}

/// Walk a directory recursively and extract comments from the files within it.
fn extract_comments_from_directory(directory: &str) {
    let mut comments = serde_json::Value::default();
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path().to_str().unwrap();
        let metadata = std::fs::metadata(path).unwrap();
        if metadata.is_file() {
            let file_comments = extract_comments_from_file(path);
            if let Ok(file_comments) = file_comments {
                comments[path] = serde_json::from_str(&file_comments).unwrap();
                println!("{comments}");
            }
        }
    }
}

fn main() {
    pretty_env_logger::init();
    let args = Cli::parse();
    let input = &args.input;

    let metadata = std::fs::metadata(input).unwrap();
    if metadata.is_file() {
        // We use unwrap below since we want the program to complain loudly if only a single file
        // is given as input and the comment extraction fails.
        // In contrast, when we extract comments from a directory, we do not want the program to
        // complain for every unprocessable file.
        let comments = extract_comments_from_file(input).unwrap();
        println!("{comments}");
    } else if metadata.is_dir() {
        extract_comments_from_directory(input)
    }
}
