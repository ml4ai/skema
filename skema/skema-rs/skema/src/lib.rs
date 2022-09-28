// Stub for SKEMA library
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use serde_json;
use serde_json::Value; // for json
use std::string::ToString;

/******** AST for the Gromet Data Structure ********/
#[derive(Deserialize, Serialize, Debug)]
#[serde(rename_all = "UPPERCASE")] // Allows variants to match to uppercase json values
#[derive(strum_macros::Display)] // Allows variants to be printed as strings if needed
pub enum FnType {
    Fn,
    Import,
}

#[derive(Deserialize, Serialize, Debug)]
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

#[derive(Deserialize, Serialize, Debug)]
pub struct ValueL {
    pub value_type: String, // could be enum?
    #[serde(deserialize_with = "de_value")]
    #[serde(serialize_with = "se_value")]
    pub value: String, // This is the generic problem. floats are exported as ints but rust exports as floats, making full generic isn't feasible since we don't know the number of instances before hand. So we import as a string to capture the data regardless of type.
}

#[derive(Deserialize, Serialize, Debug)]
pub struct GrometBox {
    pub function_type: FunctionType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<ValueL>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize, Debug)]
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

#[derive(Deserialize, Serialize, Debug)]
pub struct GrometWire {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub src: u8,
    pub tgt: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize, Debug)]
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
#[derive(Deserialize, Serialize, Debug)]
pub struct GrometBoxConditional {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<i32>, // This exist is CHIME v2, but not in documentation...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_if: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_else: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize, Debug)]
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

#[derive(Deserialize, Serialize, Debug)]
pub struct Attributes {
    #[serde(rename = "type")]
    pub r#type: FnType,
    pub value: FunctionNet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Provenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Files {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_type: Option<String>, // Could be enum?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>, // only in highest meta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub global_reference_id: Option<String>, // only in highest meta
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<Files>>, // only in highest meta
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

#[derive(Deserialize, Serialize, Debug)]
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
// Currently only for numerical values, as gromets develope will need maintaince. Ideally we could run match control flow to determine the type and how to parse, similar to the deserialization for this field.
fn se_value<S>(x: &str, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // This is an interim solution for numerics, non numerics will require custom serialization for entire ValueL struct.
    let mut parse_f32: f32 = x.parse().unwrap();

    if parse_f32 == parse_f32.round() {
        // if the float can be truncated without percision loss, truncate
        s.serialize_i32(parse_f32 as i32)
    } else {
        // else include up to 7 digits for now, as per GroMEt specs
        parse_f32 = (parse_f32 * 10_000_000.0).round() / 10_000_000.0;
        s.serialize_f32(parse_f32)
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_roundtrip_serialization(path_example: &str) -> () {
        let mut file_contents = fs::read_to_string(path_example).expect("Unable to read file");

        let res: Gromet = serde_json::from_str(&file_contents).expect("Unable to parse");
        let res_serialized = serde_json::to_string(&res).unwrap();

        // processing the imported data
        file_contents = file_contents.replace("\n", "").replace(" ", "");

        assert_eq!(res_serialized, file_contents);
    }

    #[test]
    fn de_ser_exp0() {
        test_roundtrip_serialization("../../../data/gromet/examples/exp0/exp0--Gromet-FN-auto.json");
    }

    #[test]
    fn de_ser_exp2() {
        test_roundtrip_serialization("../../../data/gromet/examples/exp2/exp2--Gromet-FN-auto.json");
    }

    #[test]
    fn de_ser_fun3() {
        test_roundtrip_serialization("../../../data/gromet/examples/fun3/fun3--Gromet-FN-auto.json");
    }

    #[test]
    fn de_ser_while2() {
        test_roundtrip_serialization("../../../data/gromet/examples/while2/while2--Gromet-FN-auto.json");
    }

    #[test]
    fn de_ser_chime() {
        let path_example = "../../../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.2/CHIME_SIR_while_loop--Gromet-FN-auto_v2.json";
        let mut data = fs::read_to_string(path_example).expect("Unable to read file");

        let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");
        let mut res_serialized = serde_json::to_string(&res).unwrap();

        // processing serialized data
        res_serialized = res_serialized.replace("\n", "");
        res_serialized = res_serialized.replace(" ", "");

        // processing the imported data
        data = data.replace("\n", "");
        data = data.replace(" ", "");

        assert_eq!(res_serialized, data);
    }
}
