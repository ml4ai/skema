use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs::File;
use std::io::{self, BufRead};
use std::error::Error;
use serde_json;

#[derive(Default, Debug, Deserialize, Serialize)]
struct Comments {
    #[serde(rename = "$file_head")]
    file_head: Option<Vec<String>>,

    #[serde(rename = "$file_head")]
    file_foot: Option<Vec<String>>
}

fn line_is_comment(line: &String) -> bool {
    line.starts_with("!")
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
            if (line_is_comment(&l) || l.trim() == "") {
                curr_comment.push(l)
            }
            else {
                if let None = comments.file_head {
                    comments.file_head = Some(curr_comment.clone())
                }
            }
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
    let comments = get_comments("../../data/epidemiology/CHIME/CHIME_SIR_model/code/CHIME_SIR.for".to_string());
}
