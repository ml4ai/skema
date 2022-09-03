// Stub for SKEMA library
use serde::{Deserialize, Serialize};
use strum_macros::Display; // used for macro on enums

use std::string::ToString; // used for macro on enums

/******** AST for the Gromet Data Structure ********/
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")] // Allows variants to match to uppercase json values
#[derive(strum_macros::Display)] // Allows variants to be printed as strings if needed
pub enum FnType {
    Fn,
    Import,
}

#[derive(Deserialize, Serialize)]
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

#[derive(Deserialize, Serialize)]
pub struct Value {
    pub value_type: String, // could be enum?
    pub value: f32,         // This is the generic problem
}

#[derive(Deserialize, Serialize)]
pub struct GrometBox {
    pub function_type: FunctionType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize)]
pub struct GrometPort {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u8>,
    #[serde(rename = "box")]
    pub r#box: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize)]
pub struct GrometWire {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub src: u8,
    pub tgt: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize)]
pub struct GrometBoxLoop {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize)]
pub struct GrometBoxConditional {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_if: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_else: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize)]
pub struct FunctionNet {
    pub b: [GrometBox; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opi: Option<Vec<GrometPort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub opo: Option<Vec<GrometPort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wopio: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bf: Option<Vec<GrometBox>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pif: Option<Vec<GrometPort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pof: Option<Vec<GrometPort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wfopi: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wfl: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wff: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wfc: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wfopo: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bl: Option<Vec<GrometBoxLoop>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pil: Option<Vec<GrometPort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pol: Option<Vec<GrometPort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wlopi: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wll: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wlf: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wlc: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wlopo: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wl_iiargs: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wl_ioargs: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wl_cargs: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bc: Option<Vec<GrometBoxConditional>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pic: Option<Vec<GrometPort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub poc: Option<Vec<GrometPort>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wcopi: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wcl: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wcf: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wcc: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wcopo: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wc_cargs: Option<Vec<GrometWire>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize)]
pub struct Attributes {
    #[serde(rename = "type")]
    pub r#type: FnType,
    pub value: FunctionNet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize)]
pub struct Provenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_type: Option<String>, // Could be enum?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_language: Option<String>, // Could be enum?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_language_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_type: Option<String>, // Could be enum?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_file_reference_uid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_begin: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_end: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub col_begin: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub col_end: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
}

#[derive(Deserialize, Serialize)]
pub struct Gromet {
    pub name: String,
    #[serde(rename = "fn")]
    pub r#fn: FunctionNet,
    pub attributes: Vec<Attributes>,
    #[serde(skip_serializing_if = "Option::is_none")]
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
