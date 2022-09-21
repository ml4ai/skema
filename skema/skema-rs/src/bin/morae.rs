use serde_json; // for json

use std::env;
use std::fs;

use skema::FunctionType;
use skema::Gromet; // This brings in the Gromet Data struct
                   /* TODO: Write up latex documentation in docs. */

// ../../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.2/CHIME_SIR_while_loop--Gromet-FN-auto_v2_labeled.json
#[derive(Debug, Clone)]
pub struct Subroutine {
    pub name: String,
    pub indexes: Vec<u32>,
    pub att_idx: u32,
    pub label: Option<Vec<String>>,
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

    let mut subroutines: Vec<Subroutine> = vec![];
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
                        }
                        // if predicate grab indexes
                        FunctionType::Predicate => {
                            idxs.push(boxf.contents.unwrap().try_into().unwrap());
                        }
                        // if function grab indexes
                        FunctionType::Function => {
                            idxs.push(boxf.contents.unwrap().try_into().unwrap());
                        }
                        // else do nothing
                        _ => {}
                    }
                }
                // define a new instance of UserFunction
                if entry.value.b[0].name.as_ref().is_none() {
                    let uf = Subroutine {
                        // name, un-named
                        name: String::from("un-named"),
                        // pull from idxs list made
                        indexes: idxs,
                        att_idx: idx,
                        label: None,
                    };
                    subroutines.push(uf);
                } else {
                    let uf = Subroutine {
                        // pull name from attribute entry
                        name: entry.value.b[0].name.as_ref().unwrap().to_string(),
                        // pull from idxs list made
                        indexes: idxs,
                        att_idx: idx,
                        label: None,
                    };
                    subroutines.push(uf);
                }
            }
            _ => {}
        }
        idx += 1;
    }

    // we now need to perform a contraction on the list of functions/subroutines. The scale on contraction is up to us, and we will choose to contract to named functions. Further contraction would likely just lead to the main function with everything called.

    // iterate through subroutines, need to use indicies since iterators create unmutable references.
    // this one will work for main-last type programs, namely the dependencies come first. (starts in reverse order)

    for i in (0..subroutines.len()).rev() {
        // find un-named subroutines
        if subroutines[i].name == String::from("un-named") {
            // intialize a counter
            // iterate through other functions
            for j in 0..subroutines.len() {
                // iterate through the functions dependencies
                for k in 0..subroutines[j].indexes.len() {
                    // if the function depends on the un-named function
                    if subroutines[j].indexes[k] == subroutines[i].att_idx {
                        // append the un-named functions dependencies
                        let mut idx_temp = subroutines[i].indexes.to_vec();
                        subroutines[j].indexes.append(&mut idx_temp);
                    }
                }
            }
        }
    }

    // clone, to allow use of iterator
    let mut subroutines_contracted = subroutines.to_vec();
    let mut unnamed: Vec<u32> = vec![];

    let mut m = 0;
    for func in subroutines.iter() {
        if func.name == String::from("un-named") {
            unnamed.push(m);
        }
        m += 1;
    }

    for l in unnamed.iter().rev() {
        subroutines_contracted.remove((*l).try_into().unwrap());
    }
    // we now perform a check to make sure every entry in attributes has been captured and there are no duplicates
    let mut tot = 1; // This starting at 1 is because main is not counted as a dependency
    for i in 0..subroutines_contracted.len() {
        subroutines_contracted[i].indexes.sort();
        subroutines_contracted[i].indexes.dedup();
        tot += subroutines_contracted[i].indexes.len();
    }
    // we now label each function, for coarse grain labeling
    if tot == res.attributes.len() {
        println!("All attributes accounted for.");
    } else {
        println!(
            "ERROR: Missing Attributes! \n {}, {}",
            tot,
            res.attributes.len()
        );
    }

    for i in 0..subroutines_contracted.len() {
        if subroutines_contracted[i].name == "main" {
            subroutines_contracted[i].label =
                Some(vec![String::from("MODEL"), String::from("PREPROCESS")]);
        } else if subroutines_contracted[i].name == "sir" {
            subroutines_contracted[i].label = Some(vec![String::from("MODEL")]);
        } else if subroutines_contracted[i].name == "sim_sir" {
            subroutines_contracted[i].label = Some(vec![String::from("MODEL")]);
        } else {
            subroutines_contracted[i].label = Some(vec![String::from("PREPROCESS")]);
        }
    }

    println!("{:?}", subroutines_contracted);
    // let res_serialized = serde_json::to_string_pretty(&res).unwrap();
}
