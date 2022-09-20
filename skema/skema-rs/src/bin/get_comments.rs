use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

//mod skema::fortran_syntax;
use skema::fortran_syntax::{line_ends_subpgm, line_is_comment, line_starts_subpgm};

// TODO: Implement support for internal comments
#[derive(Default, Debug, Deserialize, Serialize)]
struct SubprogramComments {
    head: Vec<String>,
    neck: Vec<String>,
    foot: Vec<String>,
}

#[derive(Default, Debug, Deserialize, Serialize)]
struct Comments {
    #[serde(rename = "$file_head")]
    file_head: Option<Vec<String>>,

    #[serde(rename = "$file_foot")]
    file_foot: Option<Vec<String>>,
    subprograms: HashMap<String, SubprogramComments>,
}

fn get_comments(src_file_name: String) -> Result<Comments, Box<dyn Error + 'static>> {
    let mut curr_comment: Vec<String> = Vec::new();
    let mut curr_fn: Option<String> = None;
    let mut prev_fn: Option<String> = None;
    let mut curr_marker: Option<String> = None;
    let mut in_neck = false;
    let mut comments = Comments::default();
    let extension = Path::new(&src_file_name).extension();
    let f = File::open(&src_file_name)?;
    let lines = io::BufReader::new(f).lines();

    for line in lines {
        if let Ok(l) = line {
            if line_is_comment(&l) {
                curr_comment.push(l.clone())
            } else {
                if let None = comments.file_head {
                    comments.file_head = Some(curr_comment.clone())
                }
            }

            let (f_start, f_name_maybe) = line_starts_subpgm(&l);

            if f_start {
                let f_name = f_name_maybe;
                let prev_fn = curr_fn.clone();
                let curr_fn = f_name;

                if let Some(x) = prev_fn {
                    comments
                        .subprograms
                        .get_mut(&x)
                        .expect("Subprogram named {x} not found in comment dictionary!")
                        .foot = curr_comment.clone();
                }

                comments.subprograms.insert(
                    curr_fn.unwrap(),
                    SubprogramComments {
                        head: curr_comment.clone(),
                        neck: Vec::new(),
                        foot: Vec::new(),
                    },
                );

                curr_comment = Vec::new();
                in_neck = true;
            } else if line_ends_subpgm(&l) {
                curr_comment = Vec::new();
            }
            //else if line_is_continuation(&l, extension) {

            //}
        }
    }

    println!("{}", serde_json::to_string_pretty(&comments).unwrap());
    Ok(comments)
}

fn main() {
    println!("{}", "hello")
}

#[test]
fn test_get_comments() {
    let comments = get_comments(
        "../../data/epidemiology/CHIME/CHIME_SIR_model/code/CHIME_SIR.for".to_string(),
    );
}
