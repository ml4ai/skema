use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Serialize, Deserialize)]
pub struct Specie {
    pub sname: String,
    pub uid: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Transition {
    pub tname: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputArc {
    pub it: usize,
    pub is: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputArc {
    pub ot: usize,
    pub os: usize,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ACSet {
    pub S: Vec<Specie>,
    pub T: Vec<Transition>,
    pub I: Vec<InputArc>,
    pub O: Vec<OutputArc>,
}
