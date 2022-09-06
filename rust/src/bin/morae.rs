use serde_json; // for json

use std::fs;

use skema::Gromet; // This brings in the Gromet Data struct

/* TODO: Process data for generic type post deserialization. */
/* TODO: Reserialize to make sure ingesting correctly. Including making tests.  */
/* TODO: Write up latex documentation in docs. */
fn main() {
    let path_example = "../data/gromet/examples/exp0/exp0--Gromet-FN-auto.json";
    // let path_example = "../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.2/CHIME_SIR_while_loop--Gromet-FN-auto_v2.json";

    let mut data = fs::read_to_string(path_example).expect("Unable to read file");

    let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");

    let res_serialized = serde_json::to_string(&res).unwrap();

    // processing the imported data
    data = data.replace("\n", "");
    data = data.replace(" ", "");

    // println!("{} \n{}", res_serialized, data);

    assert_eq!(res_serialized, data);
    // println!("{}", res.r#fn.b[0].function_type)
}
