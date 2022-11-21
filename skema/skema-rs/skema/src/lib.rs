// Inclusion of additional modules
pub mod gromet_memgraph;
pub mod services;

// Stub for SKEMA library
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

use serde_json::Value; // for json
use std::string::ToString;

/******** AST for the Gromet Data Structure ********/
#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "UPPERCASE")] // Allows variants to match to uppercase json values
#[derive(strum_macros::Display)] // Allows variants to be printed as strings if needed
pub enum FnType {
    Fn,
    Import,
}

#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
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

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ValueL {
    pub value_type: String, // could be enum?
    #[serde(deserialize_with = "de_value")]
    #[serde(serialize_with = "se_value")]
    pub value: String, // This is the generic problem. floats are exported as ints but rust exports as floats, making full generic isn't feasible since we don't know the number of instances before hand. So we import as a string to capture the data regardless of type.
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct GrometBox {
    pub function_type: FunctionType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<ValueL>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct GrometPort {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(rename = "box")]
    pub r#box: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<u32>, // pof: 473, 582, b: 685, 702, 719, 736, most b's
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct GrometWire {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub src: u8,
    pub tgt: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct GrometBoxLoop {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<u32>,
}
// condition, body_if, and body_else don't match online documentation
// They are vecs of gromet boxes not integers...
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct GrometBoxConditional {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<u32>, // This exist is CHIME v2, but not in documentation...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_if: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body_else: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<u32>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
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

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Attribute {
    #[serde(rename = "type")]
    pub r#type: FnType,
    pub value: FunctionNet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Vec<Metadata>>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Provenance {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Files {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uid: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_type: Option<String>, // Could be enum?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gromet_version: Option<String>,
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

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Gromet {
    pub schema: String,
    pub schema_version: String,
    pub name: String,
    #[serde(rename = "fn")]
    pub r#fn: FunctionNet,
    pub attributes: Vec<Attribute>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_collection: Option<Vec<Vec<Metadata>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<u32>,
}

// Methods
// This is a custom deserialization of the value field in the Value struct.
fn de_value<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    Ok(match Value::deserialize(deserializer)? {
        Value::Number(num) => num.to_string(),
        Value::Bool(boo) => boo.to_string(),
        Value::Array(vl) => {
            // need to construct an instance of the vector here then stringify it
            let vals = serde_json::to_string(&vl).unwrap();
            vals
        } // this will encode the vector as a string, re-serializing will be more difficult though
        Value::Object(map) => {
            let f = format!("{:?}", map);
            f
        } // this handles if the map is encoded as a map, make sure this still works with character matching..
        Value::String(strng) => {
            let mut it = strng.chars().peekable();
            let c = if let Some(&c) = it.peek() { c } else { '_' };
            match c {
                '{' => {
                    strng.to_string() // This handles if the map is encoded as a string
                }
                _ => {
                    format!("{:?}", strng)
                }
            }
            // Add conditional that will takes maps starting with '{' and format {} but for other strings format as {:?} to parse later as '"'
        }
        // Need to add support for List types (SVIIvR) --- This one is nontrivial...
        _ => return Err(de::Error::custom("Not Recognized Type")),
    })
}

// This is a custom serialization of the value field in the Value struct.
fn se_value<S>(x: &str, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    // Having to implement a custom parser based on the first character of the strings.
    // t | f for bools, else for numbers, should be able to extend to { for maps and [ for lists
    let mut it = x.chars().peekable(); // characterization allows for handling most edge cases, (strings named after primatives)
    let c = if let Some(&c) = it.peek() { c } else { 'x' };
    // we run a match on the first character because otherwise we would need to know every possible string if we run it on the full words.
    // 't' and 'f' are unique as characters are encoded as "t" so if there is a character "t" it will be parsed correctly
    match c {
        '[' => {
            let vals: Vec<ValueL> = serde_json::from_str(x).unwrap();
            s.collect_seq(vals.iter()) // This is to serialize a vector, WARNING: serde is only implemented for vecs up to length 32 by default.
        }
        '{' => s.serialize_str(x), // This is just if maps are serialized as strings, change if that changes
        '"' => {
            let char_vec: Vec<char> = x.chars().collect();
            s.serialize_char(char_vec[1])
        }
        't' | 'f' => {
            let parse_bool: bool = x.parse().unwrap();
            s.serialize_bool(parse_bool)
        }
        _ => {
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
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_roundtrip_serialization(path_example: &str) {
        let mut file_contents = fs::read_to_string(path_example).expect("Unable to read file");

        let res: Gromet = serde_json::from_str(&file_contents).expect("Unable to parse");
        let mut res_serialized = serde_json::to_string(&res).unwrap();

        // processing the imported data
        file_contents = file_contents.replace('\n', "").replace(' ', "");
        res_serialized = res_serialized.replace('\n', "").replace(' ', "");

        assert_eq!(res_serialized, file_contents);
    }

    #[test]
    fn de_ser_cond1() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/cond1/FN_0.1.4/cond1--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_dict1() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/dict1/FN_0.1.4/dict1--Gromet-FN-auto-meta.json",
        );
    }

    #[test]
    fn de_ser_exp0() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/exp0/FN_0.1.4/exp0--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_exp1() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/exp1/FN_0.1.4/exp1--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_exp2() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/exp2/FN_0.1.4/exp2--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_for1() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/for1/FN_0.1.4/for1--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_fun1() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/fun1/FN_0.1.4/fun1--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_fun2() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/fun2/FN_0.1.4/fun2--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_fun3() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/fun3/FN_0.1.4/fun3--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_fun4() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/fun4/FN_0.1.4/fun4--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_while1() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/while1/FN_0.1.4/while1--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_while2() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/while2/FN_0.1.4/while2--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_while3() {
        test_roundtrip_serialization(
            "../../../data/gromet/examples/while3/FN_0.1.4/while3--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_chime() {
        test_roundtrip_serialization(
            "../../../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.4/CHIME_SIR_while_loop--Gromet-FN-auto.json",
        );
    }

    #[test]
    fn de_ser_chime_sviivr() {
        test_roundtrip_serialization(
            "../../../data/epidemiology/CHIME/CHIME_SVIIvR_model/gromet/FN_0.1.4/CHIME_SVIIvR--Gromet-FN-auto-no_lists.json",
        );
    }

    #[test]
    fn de_ser_chime_sviivr_lists() {
        test_roundtrip_serialization(
            "../../../data/epidemiology/CHIME/CHIME_SVIIvR_model/gromet/FN_0.1.4/CHIME_SVIIvR--Gromet-FN-auto.json",
        );
    }
}
