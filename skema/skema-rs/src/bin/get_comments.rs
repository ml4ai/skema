use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use regex::Regex;

use lazy_static::lazy_static;

#[derive(Default, Debug, Deserialize, Serialize)]
struct Comments {
    #[serde(rename = "$file_head")]
    file_head: Option<Vec<String>>,

    #[serde(rename = "$file_foot")]
    file_foot: Option<Vec<String>>,
}


fn line_is_comment(line: &String) -> bool {
    // From FORTRAN Language Reference
    // (https://docs.oracle.com/cd/E19957-01/805-4939/z40007332024/index.html)

    // A line with a c, C, '*', d, D, or ! in column one is a comment line, except
    // that if the -xld option is set, then the lines starting with D or d are
    // compiled as debug lines. The d, D, and ! are nonstandard.

    // If you put an exclamation mark (!) in any column of the statement field,
    // except within character literals, then everything after the ! on that
    // line is a comment.

    // A totally blank line is a comment line as well.

    //let FORTRAN_COMMENT_CHAR_SET: HashSet<char> = HashSet::from(['c', 'C', 'd', 'D', '*', '!']);
    lazy_static! {
        static ref FORTRAN_COMMENT_CHAR_SET: HashSet<char> =
            HashSet::from(['c', 'C', 'd', 'D', '*', '!']);
    }

    match &line.chars().nth(0) {
        Some(c) => FORTRAN_COMMENT_CHAR_SET.contains(c),
        None => true,
    }
}

fn line_starts_subpgm(line: &String) -> (bool, Option<String>) {
    /// Indicates whether a line in the program is the first line of a subprogram
    /// definition.
    ///
    /// # Arguments
    ///
    /// * `line` - The line of code to analyze
    ///Returns:
    ///    (true, f_name) if line begins a definition for subprogram f_name;
    ///    (false, None) if line does not begin a subprogram definition.
    lazy_static! {
        static ref RE_SUB_START: Regex = Regex::new(r"\s*subroutine\s+(\w+)\s*\(").unwrap();
    }
    let captures = RE_SUB_START.captures(line);
    if let Some(c) = captures {
        println!("captures");
        println!("{}", &c[0]);
    }

    (false, Some("".to_string()))
}

//match = RE_SUB_START.match(line)
//if match is not None:
//f_name = match.group(1)
//return True, f_name

//match = RE_FN_START.match(line)
//if match is not None:
//f_name = match.group(2)
//return True, f_name

//return False, None

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
                curr_comment.push(l)
            } else {
                if let None = comments.file_head {
                    comments.file_head = Some(curr_comment.clone())
                }
            }

            //line_starts_subpgm(&l);
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
