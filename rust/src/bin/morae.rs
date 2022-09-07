use serde_json; // for json

use std::env;
use std::fs;

use skema::Gromet; // This brings in the Gromet Data struct

/* TODO: Write up latex documentation in docs. */
fn main() {
    let args: Vec<String> = env::args().collect();

    let path = &args[1];

    // let path_example = "../data/gromet/examples/exp0/exp0--Gromet-FN-auto.json";
    // let path_example = "../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.2/CHIME_SIR_while_loop--Gromet-FN-auto_v2.json";

    let mut data = fs::read_to_string(path).expect("Unable to read file");

    let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");

    let res_serialized = serde_json::to_string_pretty(&res).unwrap();

    println!("{}", res_serialized)
}
