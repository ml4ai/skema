//
//  Read files and produce a vector of lines from each, then
//  print each line
//

use nom::{
    branch::alt,
    multi::fold_many0,
    sequence::terminated,
    bytes::complete::{is_not, tag, take},
    IResult
};
use std::env;

// parse a line that has text and then a newline char
fn non_empty_line(s: &str) -> IResult<&str,&str> {
    let ret = terminated(is_not("\n"),tag("\n"))(s);
    match ret {
        Ok((_,b)) => {
            let n: usize = b.len() + 1;
            take(n)(s)
        }
        Err(_) => ret
    }
}

// parse a line that is only a newline char
fn empty_line(s: &str) -> IResult<&str,&str> {
    tag("\n")(s)
}

// return a string as a vector of lines
fn line_parser(s: &str) -> IResult<&str,Vec<&str>> {
    fold_many0(
        alt((
            non_empty_line,
            empty_line
        )),
        Vec::new,
        | mut acc: Vec<_>, item | {
            acc.push(item);
            acc
        }
    )(s)
}

// an entire file as a single string
fn process_string(s: &str) {
    let(_,lines) = line_parser(&s).unwrap();
    for line in lines {
        print!("{}", line);
    }
}

// read a file as one big string and process it
fn process_file(file_path: &str) {
    println!("Processing file {:?}",file_path);
    let string = std::fs::read_to_string(file_path);
    match string {
        Ok(s) => process_string(&s),
        Err(e) => println!("Error: {e:?}")
    }
}

// process arguments as input files
fn main() {
    let args: Vec<String> = env::args().collect();
    let filenames = &args[1..];
    for filename in filenames {
        process_file(filename);
    }
}

