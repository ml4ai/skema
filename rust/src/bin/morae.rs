use serde::Deserialize;
use serde_json; // for json
use strum_macros::Display;

use std::fs;
use std::string::ToString; // for file system

/* TODO: Recursive Function Netoworks? */
/* TODO: Process data for generic type post deserialization. */

fn main() {
    // Let's make a simple strongly typed data structure for this simplest gromet to get used to the notation and semantics.
    // also fn is a key word in rust so have to mitigate that for the deserialization.

    #[derive(Deserialize)]
    #[serde(rename_all = "UPPERCASE")] // Allows variants to match to uppercase json values
    #[derive(strum_macros::Display)] // Allows variants to be printed as strings if needed
    enum FnType {
        Fn,
        Import,
    }

    #[derive(Deserialize)]
    #[serde(rename_all = "UPPERCASE")]
    #[derive(strum_macros::Display)]
    enum FunctionType {
        Function,
        Predicate,
        Primitive,
        Module,
        Expression,
        Literal,
    }

    #[derive(Deserialize)]
    struct Value {
        value_type: String, // could be enum?
        value: f32,         // This is the generic problem
    }

    #[derive(Deserialize)]
    struct GrometBox {
        function_type: FunctionType,
        name: Option<String>,
        contents: Option<i32>,
        value: Option<Value>,
        metadata: Option<Vec<Metadata>>,
    }

    #[derive(Deserialize)]
    struct GrometPort {
        name: Option<String>,
        id: Option<u8>,
        #[serde(rename = "box")]
        r#box: u8,
        metadata: Option<Vec<Metadata>>,
    }

    #[derive(Deserialize)]
    struct GrometWire {
        name: Option<String>,
        src: i32,
        tgt: i32, // These can be negative?? Not natural numbers?
        metadata: Option<Vec<Metadata>>,
    }

    #[derive(Deserialize)]
    struct GrometBoxLoop {
        name: Option<String>,
        condition: Option<i32>,
        init: Option<i32>,
        body: Option<i32>,
        metadata: Option<Vec<Metadata>>,
    }

    #[derive(Deserialize)]
    struct GrometBoxConditional {
        name: Option<String>,
        condition: Option<i32>,
        body_if: Option<i32>,
        body_else: Option<i32>,
        metadata: Option<Vec<Metadata>>,
    }

    #[derive(Deserialize)]
    struct FunctionNet {
        b: [GrometBox; 1],
        opi: Option<Vec<GrometPort>>,
        opo: Option<Vec<GrometPort>>,
        wopio: Option<Vec<GrometWire>>,
        bf: Option<Vec<GrometBox>>,
        pif: Option<Vec<GrometPort>>,
        pof: Option<Vec<GrometPort>>,
        wfopi: Option<Vec<GrometWire>>,
        wfl: Option<Vec<GrometWire>>,
        wff: Option<Vec<GrometWire>>,
        wfc: Option<Vec<GrometWire>>,
        wfopo: Option<Vec<GrometWire>>,
        bl: Option<Vec<GrometBoxLoop>>,
        pil: Option<Vec<GrometPort>>,
        pol: Option<Vec<GrometPort>>,
        wlopi: Option<Vec<GrometWire>>,
        wll: Option<Vec<GrometWire>>,
        wlf: Option<Vec<GrometWire>>,
        wlc: Option<Vec<GrometWire>>,
        wlopo: Option<Vec<GrometWire>>,
        wl_iiargs: Option<Vec<GrometWire>>,
        wl_ioargs: Option<Vec<GrometWire>>,
        wl_cargs: Option<Vec<GrometWire>>,
        bc: Option<Vec<GrometBoxConditional>>,
        pic: Option<Vec<GrometPort>>,
        poc: Option<Vec<GrometPort>>,
        wcopi: Option<Vec<GrometWire>>,
        wcl: Option<Vec<GrometWire>>,
        wcf: Option<Vec<GrometWire>>,
        wcc: Option<Vec<GrometWire>>,
        wcopo: Option<Vec<GrometWire>>,
        wc_cargs: Option<Vec<GrometWire>>,
        metadata: Option<Vec<Metadata>>,
    }

    #[derive(Deserialize)]
    struct Attributes {
        #[serde(rename = "type")]
        r#type: FnType,
        value: FunctionNet,
        metadata: Option<Vec<Metadata>>,
    }

    #[derive(Deserialize)]
    struct Provenance {
        method: Option<String>,
        timestamp: Option<String>,
    }

    #[derive(Deserialize)]
    struct Metadata {
        metadata_type: Option<String>,   // Could be enum?
        source_language: Option<String>, // Could be enum?
        source_language_version: Option<String>,
        data_type: Option<String>, // Could be enum?
        code_file_reference_uid: Option<String>,
        line_begin: Option<u32>,
        line_end: Option<u32>,
        col_begin: Option<u32>,
        col_end: Option<u32>,
        provenance: Provenance,
    }

    #[derive(Deserialize)]
    struct Gromet {
        name: String,
        #[serde(rename = "fn")]
        r#fn: FunctionNet,
        attributes: Option<Vec<Attributes>>,
        metadata: Option<Vec<Metadata>>,
    }

    // let path_example = "../data/gromet/examples/while2/while2--Gromet-FN-auto.json";
    let path_example = "../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.2/CHIME_SIR_while_loop--Gromet-FN-auto.json";
    let data = fs::read_to_string(path_example).expect("Unable to read file");
    let res: Gromet = serde_json::from_str(&data).expect("Unable to parse"); // right now f32 works for all value types we currently have..
    println!("{}", res.r#fn.b[0].function_type) // The fact these are 1 dimensional arrays makes accessing attributes ugly and more convolved than it should be.
}
