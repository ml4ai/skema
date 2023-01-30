use log::warn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use crate::languages::fortran::{
    line_ends_subpgm, line_is_comment, line_is_continuation, line_starts_subpgm,
};

// TODO: Implement support for internal comments
#[derive(Default, Debug, Deserialize, Serialize)]
struct SubprogramComments {
    head: Vec<String>,
    neck: Vec<String>,
    foot: Vec<String>,
}

#[derive(Default, Debug, Deserialize, Serialize)]
pub struct Comments {
    #[serde(rename = "$file_head")]
    file_head: Vec<String>,

    #[serde(rename = "$file_foot")]
    file_foot: Vec<String>,
    subprograms: HashMap<String, SubprogramComments>,
}

pub fn get_comments(src_file_name: &str) -> Result<Comments, Box<dyn Error + 'static>> {
    let mut curr_comment: Vec<String> = Vec::new();
    let mut curr_fn: Option<String> = None;
    let mut prev_fn: Option<String>;
    let mut in_neck = false;
    let mut comments = Comments::default();
    let extension = Path::new(&src_file_name)
        .extension()
        .expect("Unable to get extension for {src_file_name}!")
        .to_str()
        .expect("Unable to convert extension to a valid string!");
    let f = File::open(&src_file_name)?;
    let lines = io::BufReader::new(f).lines();

    for line in lines {
        if let Ok(l) = line {
            if line_is_comment(&l) {
                curr_comment.push(l)
            } else {
                if comments.file_head.is_empty() {
                    comments.file_head = curr_comment.clone()
                }
                let (f_start, f_name) = line_starts_subpgm(&l);

                if f_start {
                    prev_fn = curr_fn.clone();
                    curr_fn = f_name;

                    if let Some(x) = &prev_fn {
                        comments.subprograms.get_mut(x).unwrap().foot = curr_comment.clone();
                    }

                    comments.subprograms.insert(
                        curr_fn.as_ref().unwrap().to_string(),
                        SubprogramComments {
                            head: curr_comment.clone(),
                            neck: Vec::new(),
                            foot: Vec::new(),
                        },
                    );

                    curr_comment.clear();
                    in_neck = true;
                } else if line_ends_subpgm(&l) {
                    curr_comment.clear();
                } else if line_is_continuation(&l, &extension) {
                    continue;
                } else {
                    if in_neck {
                        comments
                            .subprograms
                            .get_mut(&curr_fn.clone().unwrap())
                            .unwrap()
                            .neck = curr_comment.clone();
                        in_neck = false;
                        curr_comment.clear();
                    }
                    // TODO: Implement logic for collecting the internal comments.
                }
            }
        }
    }

    // If there's a comment at the very end of the file, make it the foot
    // comment of curr_fn
    match curr_fn {
        None => warn!("curr_fn is None, we do not know how to handle this case!"),
        Some(c) => {
            if !curr_comment.is_empty() {
                if let Some(x) = comments.subprograms.get_mut(&c) {
                    x.foot = curr_comment.clone();
                    comments.file_foot = curr_comment;
                }
            }
        }
    }

    Ok(comments)
}

#[test]
fn test_get_comments() {
    get_comments("../../../data/epidemiology/CHIME/CHIME_SIR_model/code/CHIME_SIR.for").unwrap();
}
