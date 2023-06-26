//! Program to get comments from a source code file.

use clap::{Parser, ValueEnum};
use std::{collections::HashSet, fs::read_to_string};

use lazy_static::lazy_static;
use serde_json as json;
use std::path::Path;

use comment_extraction::{
    comments::Comment,
    conventions::dssat::get_comments as get_dssat_comments,
    languages::{
        cpp::get_comments as get_cpp_comments, fortran::line_is_comment,
        python::get_comments as get_python_comments,
    },
};

/// Some codebases (e.g., the codebase for the DSSAT crop modeling
/// system) follow a particular commenting convention. In such cases, we may want to handle the
/// comment extraction differently in order to facilitate alignment with the literature. This enum
/// contains the different conventions we handle.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, ValueEnum)]
enum Convention {
    DSSAT,
}

/// Command line arguments
#[derive(Parser)]
struct Cli {
    /// Path to input source code file
    input: String,

    /// Convention that the codebase follows (optional).
    #[arg(long, value_enum)]
    convention: Option<Convention>,
}

/// Programming language
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
    lazy_static! {
        static ref FORTRAN_EXTENSIONS: HashSet<&'static str> =
            HashSet::from(["f", "for", "F", "F70"]);
        static ref CPP_C_EXTENSIONS: HashSet<&'static str> =
            HashSet::from(["c", "cc", "cpp", "h", "hpp"]);
    }

    let extension = Path::new(filepath)
        .extension()
        .unwrap_or_else(|| panic!("Unable to get extension for {}!", filepath))
        .to_str()
        .expect("Unable to convert extension to a valid string!");
    if FORTRAN_EXTENSIONS.contains(extension) {
        Language::Fortran
    } else if extension == "py" {
        Language::Python
    } else if CPP_C_EXTENSIONS.contains(extension) {
        Language::CppOrC
    } else {
        Language::Other
    }
}

/// Extract comments from file and return them.
fn extract_comments_from_file(
    filepath: &str,
    convention: &Option<Convention>,
) -> Result<String, &'static str> {
    let language = infer_language(filepath);
    let comments: String;
    match language {
        Language::Fortran => {
            if let Some(Convention::DSSAT) = convention {
                comments = json::to_string(&get_dssat_comments(filepath).unwrap()).unwrap();
            } else {
                let fortran = read_to_string(filepath).expect("Unable to read file");
                let mut comment_vec = Vec::new();

                for (num, line) in fortran.lines().enumerate() {
                    if line_is_comment(line) {
                        let comment = Comment {
                            line_number: num + 1,
                            contents: line.trim().to_string(),
                        };
                        comment_vec.push(comment);
                    }
                }
                comments = json::to_string(&comment_vec).unwrap();
            }
        }
        Language::Python => {
            comments = json::to_string(&get_python_comments(filepath)).unwrap();
        }
        Language::CppOrC => {
            comments = json::to_string(&get_cpp_comments(filepath)).unwrap();
        }
        Language::Other => return Err(
            "File extension does not correspond to any of the ones the comment extractor handles.",
        ),
    }

    Ok(comments)
}

/// Walk a directory recursively and extract comments from the files within it.
fn extract_comments_from_directory(directory: &str, convention: &Option<Convention>) {
    let mut comments = json::Value::default();
    for entry in walkdir::WalkDir::new(directory)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path().to_str().unwrap();
        let metadata = std::fs::metadata(path).unwrap();
        if metadata.is_file() {
            let file_comments = extract_comments_from_file(path, convention);
            if let Ok(file_comments) = file_comments {
                comments[path] = json::from_str(&file_comments).unwrap();
            }
        }
    }
    println!("{comments}");
}

fn main() {
    pretty_env_logger::init();
    let args = Cli::parse();
    let input = &args.input;
    let convention = &args.convention;

    let metadata = std::fs::metadata(input).unwrap();
    if metadata.is_file() {
        // We use unwrap below since we want the program to complain loudly if only a single file
        // is given as input and the comment extraction fails.
        // In contrast, when we extract comments from a directory, we do not want the program to
        // complain for every unprocessable file.
        let comments = extract_comments_from_file(input, convention).unwrap();
        println!("{comments}");
    } else if metadata.is_dir() {
        extract_comments_from_directory(input, convention)
    }
}

/// Test DSSAT-style comment extraction
#[test]
fn test_dssat_style_extraction() {
    let output: json::Value =
        json::from_str(&read_to_string("tests/data/chime_sir_output.json").unwrap()).unwrap();
    let comments: json::Value = json::from_str(
        &json::to_string(
            &get_dssat_comments(
                "./../../../data/epidemiology/CHIME/CHIME_SIR_model/code/CHIME_SIR.for",
            )
            .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(output, comments);
}
