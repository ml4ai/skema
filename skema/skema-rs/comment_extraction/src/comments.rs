use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A struct to represent a comment.
#[derive(Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Comment {
    /// Line number of the comment.
    pub line_number: usize,

    /// Contents of the comment.
    pub contents: String,
}

#[derive(Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct Comments {
    /// Single and multi-line comments not associated with a function or class.
    pub comments: Vec<Comment>,

    /// Docstrings
    pub docstrings: HashMap<String, Vec<String>>,
}
