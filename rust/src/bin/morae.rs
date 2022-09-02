use serde_json; // for json

use std::fs;

use skema::Gromet; // This brings in the Gromet Data struct

/* TODO: Recursive Function Netoworks? */
/* TODO: Process data for generic type post deserialization. */

fn main() {
    // let path_example = "../data/gromet/examples/while2/while2--Gromet-FN-auto.json";
    let path_example = "../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.2/CHIME_SIR_while_loop--Gromet-FN-auto.json";
    let data = fs::read_to_string(path_example).expect("Unable to read file");
    let res: Gromet = serde_json::from_str(&data).expect("Unable to parse"); // right now f32 works for all value types we currently have..
    println!("{}", res.r#fn.b[0].function_type) // The fact these are 1 dimensional arrays makes accessing attributes ugly and more convolved than it should be.
}
