use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Specie {
    pub sname: String,
    pub uid: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub tname: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputArc {
    pub it: usize,
    pub is: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputArc {
    pub ot: usize,
    pub os: usize,
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ACSet {
    pub S: Vec<Specie>,
    pub T: Vec<Transition>,
    pub I: Vec<InputArc>,
    pub O: Vec<OutputArc>,
}
