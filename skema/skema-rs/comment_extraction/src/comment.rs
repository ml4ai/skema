use serde::{Deserialize, Serialize};

/// A struct to represent a comment.
#[derive(Serialize, Deserialize)]
pub struct Comment {
    pub line_number: usize,
    pub contents: String,
}
