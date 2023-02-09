//! Structs to represent elements of ACSets (Annotated C-Sets, a concept from category theory).
//! JSON-serialized ACSets are the form of model exchange between TA1 and TA2.
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize)]
pub struct Specie {
    pub sname: String,
    pub uid: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub tname: String,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize)]
pub struct InputArc {
    pub it: usize,
    pub is: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize)]
pub struct OutputArc {
    pub ot: usize,
    pub os: usize,
}

#[allow(non_snake_case)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Serialize, Deserialize, Default)]
pub struct ACSet {
    pub S: Vec<Specie>,
    pub T: Vec<Transition>,
    pub I: Vec<InputArc>,
    pub O: Vec<OutputArc>,
}
