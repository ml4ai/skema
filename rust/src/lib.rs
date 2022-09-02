// Stub for SKEMA library
use serde::Deserialize;
use strum_macros::Display; // used for macro on enums

use std::string::ToString; // used for macro on enums

// AST for the Gromet Data Structure
#[derive(Deserialize)]
#[serde(rename_all = "UPPERCASE")] // Allows variants to match to uppercase json values
#[derive(strum_macros::Display)] // Allows variants to be printed as strings if needed
pub enum FnType {
    Fn,
    Import,
}

#[derive(Deserialize)]
#[serde(rename_all = "UPPERCASE")]
#[derive(strum_macros::Display)]
pub enum FunctionType {
    Function,
    Predicate,
    Primitive,
    Module,
    Expression,
    Literal,
}

#[derive(Deserialize)]
pub struct Value {
    pub value_type: String, // could be enum?
    pub value: f32,         // This is the generic problem
}

#[derive(Deserialize)]
pub struct GrometBox {
    pub function_type: FunctionType,
    pub name: Option<String>,
    pub contents: Option<i32>,
    pub value: Option<Value>,
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize)]
pub struct GrometPort {
    pub name: Option<String>,
    pub id: Option<u8>,
    #[serde(rename = "box")]
    pub r#box: u8,
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize)]
pub struct GrometWire {
    pub name: Option<String>,
    pub src: i32,
    pub tgt: i32, // These can be negative?? Not natural numbers?
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize)]
pub struct GrometBoxLoop {
    pub name: Option<String>,
    pub condition: Option<i32>,
    pub init: Option<i32>,
    pub body: Option<i32>,
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize)]
pub struct GrometBoxConditional {
    pub name: Option<String>,
    pub condition: Option<i32>,
    pub body_if: Option<i32>,
    pub body_else: Option<i32>,
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize)]
pub struct FunctionNet {
    pub b: [GrometBox; 1],
    pub opi: Option<Vec<GrometPort>>,
    pub opo: Option<Vec<GrometPort>>,
    pub wopio: Option<Vec<GrometWire>>,
    pub bf: Option<Vec<GrometBox>>,
    pub pif: Option<Vec<GrometPort>>,
    pub pof: Option<Vec<GrometPort>>,
    pub wfopi: Option<Vec<GrometWire>>,
    pub wfl: Option<Vec<GrometWire>>,
    pub wff: Option<Vec<GrometWire>>,
    pub wfc: Option<Vec<GrometWire>>,
    pub wfopo: Option<Vec<GrometWire>>,
    pub bl: Option<Vec<GrometBoxLoop>>,
    pub pil: Option<Vec<GrometPort>>,
    pub pol: Option<Vec<GrometPort>>,
    pub wlopi: Option<Vec<GrometWire>>,
    pub wll: Option<Vec<GrometWire>>,
    pub wlf: Option<Vec<GrometWire>>,
    pub wlc: Option<Vec<GrometWire>>,
    pub wlopo: Option<Vec<GrometWire>>,
    pub wl_iiargs: Option<Vec<GrometWire>>,
    pub wl_ioargs: Option<Vec<GrometWire>>,
    pub wl_cargs: Option<Vec<GrometWire>>,
    pub bc: Option<Vec<GrometBoxConditional>>,
    pub pic: Option<Vec<GrometPort>>,
    pub poc: Option<Vec<GrometPort>>,
    pub wcopi: Option<Vec<GrometWire>>,
    pub wcl: Option<Vec<GrometWire>>,
    pub wcf: Option<Vec<GrometWire>>,
    pub wcc: Option<Vec<GrometWire>>,
    pub wcopo: Option<Vec<GrometWire>>,
    pub wc_cargs: Option<Vec<GrometWire>>,
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize)]
pub struct Attributes {
    #[serde(rename = "type")]
    pub r#type: FnType,
    pub value: FunctionNet,
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize)]
pub struct Provenance {
    pub method: Option<String>,
    pub timestamp: Option<String>,
}

#[derive(Deserialize)]
pub struct Metadata {
    pub metadata_type: Option<String>,   // Could be enum?
    pub source_language: Option<String>, // Could be enum?
    pub source_language_version: Option<String>,
    pub data_type: Option<String>, // Could be enum?
    pub code_file_reference_uid: Option<String>,
    pub line_begin: Option<u32>,
    pub line_end: Option<u32>,
    pub col_begin: Option<u32>,
    pub col_end: Option<u32>,
    pub provenance: Provenance,
}

#[derive(Deserialize)]
pub struct Gromet {
    pub name: String,
    #[serde(rename = "fn")]
    pub r#fn: FunctionNet,
    pub attributes: Option<Vec<Attributes>>,
    pub metadata: Option<Vec<Metadata>>,
}

// Methods
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
