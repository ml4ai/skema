// Program to get comments from a source code file.
// Original version for handling Fortran code written by Saumya Debray in Python for the AutoMATES
// project (https://ml4ai.github.io/automates)
// This Rust port was written by Adarsh Pyarelal for the SKEMA project.

use log::warn;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

use clap::Parser;

//mod skema::fortran_syntax;
use skema::fortran_syntax::{
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
struct Comments {
    #[serde(rename = "$file_head")]
    file_head: Vec<String>,

    #[serde(rename = "$file_foot")]
    file_foot: Vec<String>,
    subprograms: HashMap<String, SubprogramComments>,
}

fn get_comments(src_file_name: PathBuf) -> Result<Comments, Box<dyn Error + 'static>> {
    let mut curr_comment: Vec<String> = Vec::new();
    let mut curr_fn: Option<String> = None;
    let mut prev_fn: Option<String> = None;
    let mut curr_marker: Option<String> = None;
    let mut in_neck = false;
    let mut comments = Comments::default();
    let mut lineno: usize = 1;
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
                curr_comment.push(l.clone())
            } else {
                if comments.file_head.is_empty() {
                    comments.file_head = curr_comment.clone()
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
                    curr_fn
                        .expect("Error: curr_fn is None, we do not know how to handle this case!"),
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
            } else if line_is_continuation(&l, &extension) {
                lineno += 1;
                continue;
            } else {
                // TODO: Implement the logic for collecting the neck and internal comments. The
                // logic from the original get_comments.py script doesn't quite seem to work, but
                // perhaps it is a bug on my part - Adarsh
            }
        }
        lineno += 1;
    }

    // If there's a comment at the very end of the file, make it the foot
    // comment of curr_fn
    match curr_fn {
        None => {
            warn!("curr_fn is None, we do not know how to handle this case!");
        }
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

#[derive(Parser, Debug)]
struct Cli {
    #[clap(parse(from_os_str))]
    filepath: PathBuf
}

fn main() {
    let args = Cli::parse();
    match get_comments(args.filepath) {
        Ok(c) => {println!("{}", serde_json::to_string(&c).unwrap())},
        Err(e) => panic!("Error getting the comments")
    };
    
}

#[test]
fn test_get_comments() {
    let comments = get_comments(
        "../../data/epidemiology/CHIME/CHIME_SIR_model/code/CHIME_SIR.for".to_string(),
    );
}
