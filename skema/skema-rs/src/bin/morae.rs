use serde_json; // for json

use std::env;
use std::fs;

use skema::FunctionType;
use skema::Gromet; // This brings in the Gromet Data struct
                   /* TODO: Write up latex documentation in docs. */

// ../../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.2/CHIME_SIR_while_loop--Gromet-FN-auto_v2_labeled.json
#[derive(Debug)]
pub struct UserFunction {
    pub name: String,
    pub indexes: Vec<u32>,
    pub att_idx: u32,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let data = fs::read_to_string(path).expect("Unable to read file");
    let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");

    // let att_len = res.attributes.len();

    // let's determine all the define functions of the code and label each one either model or not at first
    // then we break down the lines of the main function to get more specific
    // then we write some custom python scripts for test examples, namely nested function to refine this as well

    let mut functions: Vec<UserFunction> = vec![];
    let mut idx = 1;
    // iterate through the attributes
    for entry in res.attributes.iter() {
        //determine the function type of each attribute box
        match entry.value.b[0].function_type {
            // if theyre a function do this
            FunctionType::Function => {
                // initial vec to collect indexes of internal expressions of the function
                let mut idxs: Vec<u32> = vec![];
                // iterate through internal boxes of function
                for boxf in entry.value.bf.as_ref().unwrap().iter() {
                    // determine function type of internal boxes
                    match boxf.function_type {
                        // if expression grab indexes
                        FunctionType::Expression => {
                            idxs.push(boxf.contents.unwrap().try_into().unwrap());
                            // need to address ownership
                        }
                        FunctionType::Function => { /* implement nested functions */ }
                        // else do nothing
                        _ => {}
                    }
                }
                // define a new instance of UserFunction
                let uf = UserFunction {
                    // pull name from attribute entry
                    name: entry.value.b[0].name.as_ref().unwrap().to_string(),
                    // pull from idxs list made
                    indexes: idxs,
                    att_idx: idx,
                };
                functions.push(uf); // need to address ownership
            }
            _ => {}
        }
        idx += 1;
    }

    println!("{:?}", functions);
    // let res_serialized = serde_json::to_string_pretty(&res).unwrap();
}
