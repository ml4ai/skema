use lazy_static::lazy_static;
use serde_json as json;
use std::{collections::HashSet, fs::read_to_string, path::Path};
use walkdir;

use crate::{
    comments::Comment,
    conventions::{dssat::get_comments as get_dssat_comments, Convention},
    languages::{
        cpp::get_comments as get_cpp_comments, fortran::line_is_comment,
        python::get_comments as get_python_comments,
    },
};

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
pub fn extract_comments_from_file(
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
pub fn extract_comments_from_directory<P: AsRef<Path>>(
    directory: P,
    convention: &Option<Convention>,
) -> json::Value {
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
    comments
}
