use serde::Deserialize;
use serde_json; // for json

use std::fs; // for file system

fn main() {
    // Let's make a simple strongly typed data structure for this simplest gromet to get used to the notation and semantics.
    // also fn is a key word in rust so have to mitigate that for the deserialization.

    #[derive(Deserialize)]
    struct Value<T> {
        value_type: String,
        value: T,
    }

    #[derive(Deserialize)]
    struct B {
        function_type: String,
        name: String,
    }

    #[derive(Deserialize)]
    struct Bf<T> {
        function_type: String,
        name: String,
        contents: Option<i32>,
        value: Option<Value<T>>,
    }

    #[derive(Deserialize)]
    struct Pof {
        name: String,
        #[serde(rename = "box")]
        block: u8,
    }

    #[derive(Deserialize)]
    struct Opo {
        name: String,
        #[serde(rename = "box")]
        block: u8,
    }

    #[derive(Deserialize)]
    struct Wfopo {
        src: u8,
        tgt: u8,
    }

    #[derive(Deserialize)]
    struct FunctionNet<T> {
        b: [B; 1],
        bf: [Bf<T>; 1],
        pof: [Pof; 1],
        opo: Option<[Opo; 1]>,
        wfopo: Option<[Wfopo; 1]>,
    }

    #[derive(Deserialize)]
    struct Attributes<T> {
        #[serde(rename = "type")]
        type_name: String,
        value: FunctionNet<T>,
    }

    #[derive(Deserialize)]
    struct Metadata {}

    #[derive(Deserialize)]
    struct Gromet<T> {
        name: String,
        #[serde(rename = "fn")]
        fnet: FunctionNet<T>,
        attributes: [Attributes<T>; 1],
        metadata: [Metadata; 0],
    }

    let path_example = "../data/gromet/examples/exp0/exp0--Gromet-FN-auto.json";
    let data = fs::read_to_string(path_example).expect("Unable to read file");
    let res: Gromet<f32> = serde_json::from_str(&data).expect("Unable to parse"); // right now f32 works for all value types we currently have..
    println!("{}", res.attributes[0].type_name) // The fact these are 1 dimensional arrays makes accessing attributes ugly and more convolved than it should be.
}
