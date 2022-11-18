use serde_json; // for json

use std::env;
use std::fs;

use skema::FunctionType;
use skema::Gromet; // This brings in the Gromet Data struct

/* TODO:
- Upgrade lib to 0.1.4 because need the "bl" for while loops which doesn't exist for 0.1.2
    #- Metadata changes
    #- Include option to read gromet version as error check as it is now present
- Setup so when loop is found it is indexed in the subroutine field
- For applying roles follow:
   - Everything before loop on main function (define this beyond its name) is role: INITIALIZATION
   - Last line in the loop is the CORE_DYNAMICS (Only for CHIME, as no post-processing)
   - Everything else in the loop is PREPROCESSING
- Roles need indexes and lines to be covered as well.
- Expand lib to take in SVIIvR model, need to deserialize list of struct
    - Test to make sure both models get labeled correctly.
- Mention code by jataware to Clay as possible example for use of DSO ontology and semantic expansion
    - Will need to be cleaned up. */

// ../../data/epidemiology/CHIME/CHIME_SIR_model/gromet/FN_0.1.2/CHIME_SIR_while_loop--Gromet-FN-auto_v2_labeled.json
#[derive(Debug, Clone)]
pub struct Role {
    pub role: String,
    pub indexes: Vec<u32>,
    pub lines: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Loop {
    pub indexes: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Subroutine {
    pub name: String,
    pub indexes: Vec<u32>,
    pub att_idx: u32,
    pub label: Option<Vec<Role>>,
    pub lines: Vec<u32>,
    pub loops: Option<Loop>,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];
    let data = fs::read_to_string(path).expect("Unable to read file");
    let res: Gromet = serde_json::from_str(&data).expect("Unable to parse");

    let _res_serial = serde_json::to_string(&res).unwrap();

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
                        // else do nothing, aka internal is a literal or primative
                        _ => {}
                    }
                }
                // define a new instance of UserFunction
                if entry.value.b[0].name.as_ref().is_none() {
                    if entry.value.bl.as_ref().is_none() {
                        let uf = Subroutine {
                            // name, un-named
                            name: String::from("un-named"),
                            // pull from idxs list made
                            indexes: idxs,
                            att_idx: idx,
                            label: None,
                            lines: vec![0],
                            loops: None,
                        };
                        subroutines.push(uf);
                    } else {
                        let lp = Loop {
                            // This will stay as just relative for now since it is ignored in current analysis
                            indexes: vec![
                                idxs[entry.value.bl.as_ref().unwrap()[0].condition.unwrap()
                                    as usize],
                                idxs[(entry.value.bl.as_ref().unwrap()[0].body.unwrap() - 1)
                                    as usize],
                            ],
                        };
                        let uf = Subroutine {
                            // name, un-named
                            name: String::from("un-named"),
                            // pull from idxs list made
                            indexes: idxs,
                            att_idx: idx,
                            label: None,
                            lines: vec![0],
                            loops: Some(lp),
                        };
                        subroutines.push(uf);
                    }
                } else {
                    // This line getter is pretty bugged, as the line references have shifted some from gromet versions and now each function only labels the line its define on.
                    // get the index of the metadata
                    let lines_ind = entry.value.b[0].metadata.as_ref().unwrap();
                    // line data is in last element of index metadata vec
                    let line_meta =
                        res.metadata_collection.as_ref().unwrap()[*lines_ind as usize].len() - 1;
                    // get the begining line number
                    let lines_b = res.metadata_collection.as_ref().unwrap()[*lines_ind as usize]
                        [line_meta as usize]
                        .line_begin;
                    // get the ending line number
                    let lines_e = res.metadata_collection.as_ref().unwrap()[*lines_ind as usize]
                        [line_meta as usize]
                        .line_end;
                    // compose line location vector
                    let lines_be = vec![lines_b.unwrap(), lines_e.unwrap()];
                    if entry.value.bl.as_ref().is_none() {
                        let uf = Subroutine {
                            // pull name from attribute entry
                            name: entry.value.b[0].name.as_ref().unwrap().to_string(),
                            // pull from idxs list made
                            indexes: idxs,
                            att_idx: idx,
                            label: None,
                            lines: lines_be,
                            loops: None,
                        };
                        subroutines.push(uf);
                    } else {
                        let lp = Loop {
                            indexes: vec![
                                idxs[entry.value.bl.as_ref().unwrap()[0].condition.unwrap()
                                    as usize],
                                idxs[entry.value.bl.as_ref().unwrap()[0].body.unwrap() as usize],
                            ], // condition is the attribute for everything inside the loop and body is first line after loop.
                        };
                        let uf = Subroutine {
                            // pull name from attribute entry
                            name: entry.value.b[0].name.as_ref().unwrap().to_string(),
                            // pull from idxs list made
                            indexes: idxs,
                            att_idx: idx,
                            label: None,
                            lines: lines_be,
                            loops: Some(lp),
                        };
                        subroutines.push(uf);
                    }
                }
            }
            _ => {}
        }
        idx += 1;
    }
    // we now need to perform a contraction on the list of functions/subroutines. The scale on contraction is up to us, and we will choose to contract to named functions. Further contraction would likely just lead to the main function with everything called.

    // iterate through subroutines, need to use indicies since iterators create unmutable references.
    // this one will work for main-last type programs, namely the dependencies come first. (starts in reverse order)
    // we lose the information about loops of un-named functions in this contraction for now.
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
    //let mut tot = 0; // This starting at 1 is because main is not counted as a dependency
    for i in 0..subroutines_contracted.len() {
        subroutines_contracted[i].indexes.sort();
        subroutines_contracted[i].indexes.dedup();
        //tot += subroutines_contracted[i].indexes.len();
    }

    // now for the implementation of the heuristics for labeling, note we have the contracted, trimmed subroutines and the raw uncontracted and untrimmed subroutines to work with.

    let mut roles: Vec<Role> = vec![];

    let name_top = &subroutines_contracted[(subroutines_contracted.len() - 1)].name;

    for i in (0..subroutines.len()).rev() {
        if *name_top == subroutines[i].name {
            if !subroutines[i].loops.is_none() {
                // collect relevant loop deps
                let loop_start = subroutines[i].loops.as_ref().unwrap().indexes[0];
                let loop_end = subroutines[i].loops.as_ref().unwrap().indexes[1];
                // collect all deps for function
                let top_dep = subroutines[i].indexes.to_vec();
                // grab deps of the loop
                let mut loop_dep: Vec<u32> = vec![];
                for j in (i + 1)..subroutines.len() {
                    if subroutines[j].att_idx == loop_start {
                        let mut idx_temp = subroutines[j].indexes.to_vec();
                        loop_dep.append(&mut idx_temp);
                    }
                }
                // constructing first label, core_dynamics
                let mut core_dep: Vec<u32> = vec![];
                // this finds all the additional deps beyond the loop end
                let mut count = 0;
                for j in (0..subroutines[i].indexes.len()).rev() {
                    if subroutines[i].indexes[j] == loop_end {
                        count = subroutines[i].indexes.len() - j;
                    }
                }
                let core_dep_count = count - loop_dep.len() - 1;
                core_dep.push(loop_end);
                if core_dep_count != 0 {
                    let bot = subroutines[i].indexes.len() - core_dep_count;
                    let mut new_deps = subroutines[i].indexes[bot..].to_vec();
                    core_dep.append(&mut new_deps);
                }
                // now to construct all the roles
                loop_dep.push(loop_start);
                let init_end = top_dep.len() - core_dep.len() - loop_dep.len();
                let core = Role {
                    role: String::from("Core_Dynamics"),
                    indexes: core_dep,
                    lines: vec![0],
                };
                let prep = Role {
                    role: String::from("Preprocessing"),
                    indexes: loop_dep,
                    lines: vec![0],
                };
                let init = Role {
                    role: String::from("Initialization"),
                    indexes: top_dep[..init_end].to_vec(),
                    lines: vec![0],
                };
                roles.push(core);
                roles.push(prep);
                roles.push(init);
            }
        }
    }

    // now we need to contract the role deps and remove duplications
    for i in (0..subroutines.len()).rev() {
        // intialize a counter
        // iterate through other functions
        for j in 0..roles.len() {
            // iterate through the functions dependencies
            for k in 0..roles[j].indexes.len() {
                // if the function depends on the un-named function
                if roles[j].indexes[k] == subroutines[i].att_idx {
                    // append the un-named functions dependencies
                    let mut idx_temp = subroutines[i].indexes.to_vec();
                    roles[j].indexes.append(&mut idx_temp);
                }
            }
        }
    }

    for i in 0..roles.len() {
        roles[i].indexes.sort();
        roles[i].indexes.dedup();
    }

    // printing results
    // println!("{:?}", subroutines);
    println!("\n{:?}\n", subroutines_contracted);
    println!("{:?}\n", roles);
    // let res_serialized = serde_json::to_string_pretty(&res).unwrap();
}
