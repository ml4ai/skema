// Stub for SKEMA library
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_json;
use serde_json::Value; // for json
use std::fs;
use std::string::ToString;
use strum_macros::Display; // used for macro on enums // used for macro on enums

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
pub struct ValueL {
    pub value_type: String, // could be enum?
    #[serde(deserialize_with = "de_value")]
    #[serde(serialize_with = "se_value")]
    pub value: String, // This is the generic problem. floats are exported as ints but rust exports as floats, making full generic isn't feasible since we don't know the number of instances before hand. So we import as a string to capture the data regardless of type.
}

#[derive(Deserialize, Serialize)]
pub struct GrometBox {
    pub function_type: FunctionType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<ValueL>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct GrometPort {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
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
// condition, body_if, and body_else don't match online documentation
// They are vecs of gromet boxes not integers...
#[derive(Deserialize, Serialize)]
pub struct GrometBoxConditional {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<Vec<GrometBox>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_if: Option<Vec<GrometBox>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_else: Option<Vec<GrometBox>>,
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
    pub index: Option<u8>,
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
// This is a custom deserialization of the value field in the Value struct.
// Currently only for numerical values, as gromets develope will need maintaince.
fn de_value<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::Number(num) => num.to_string(),
        _ => return Err(de::Error::custom("wrong type")),
    })
}

// This is a custom serialization of the value field in the Value struct.
// Currently only for numerical values (only ints right now too), as gromets develope will need maintaince.
fn se_value<S>(x: &str, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let parse_num: i32 = x.parse().unwrap();
    s.serialize_i32(parse_num)
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn de_ser_exp0() {
        let path_example = "../data/gromet/examples/exp0/exp0--Gromet-FN-auto.json";
        let mut data = fs::read_to_string(path_example).expect("Unable to read file");

        let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");
        let res_serialized = serde_json::to_string(&res).unwrap();

        // processing the imported data
        data = data.replace("\n", "");
        data = data.replace(" ", "");

        assert_eq!(res_serialized, data);
    }

    #[test]
    fn de_ser_fun3() {
        let path_example = "../data/gromet/examples/fun3/fun3--Gromet-FN-auto.json";
        let mut data = fs::read_to_string(path_example).expect("Unable to read file");

        let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");
        let res_serialized = serde_json::to_string(&res).unwrap();

        // processing the imported data
        data = data.replace("\n", "");
        data = data.replace(" ", "");

        assert_eq!(res_serialized, data);
    }

    #[test]
    fn de_ser_while2() {
        let path_example = "../data/gromet/examples/while2/while2--Gromet-FN-auto.json";
        let mut data = fs::read_to_string(path_example).expect("Unable to read file");

        let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");
        let res_serialized = serde_json::to_string(&res).unwrap();

        // processing the imported data
        data = data.replace("\n", "");
        data = data.replace(" ", "");

        assert_eq!(res_serialized, data);
    }

    #[test]
    fn de_ser_cond1() {
        let path_example = "../data/gromet/examples/cond1/cond1--Gromet-FN-auto.json";
        let mut data = fs::read_to_string(path_example).expect("Unable to read file");

        let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");
        let res_serialized = serde_json::to_string(&res).unwrap();

        // processing the imported data
        data = data.replace("\n", "");
        data = data.replace(" ", "");

        assert_eq!(res_serialized, data);
    }
}
