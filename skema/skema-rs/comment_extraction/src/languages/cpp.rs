//
//  find C and C++ comments by tag
//

use nom::{
    branch::alt,
    bytes::complete::{tag, take, take_until},
    multi::fold_many0,
    IResult
};

use std::env;

// newlines can occur within these comments
fn find_c_comment(s: &str) -> IResult<&str, &str> {
    match tag("/*")(s) {
        Ok((i,_)) => {
            match take_until("*/")(i) {
                Ok((_,o)) => {
                    let n: usize = o.len() + 4;
                    take(n)(s)
                }
                Err(e) => Err(e)
            }
        }
        Err(e) => Err(e)
    }
}

fn find_cpp_comment(s: &str) -> IResult<&str, &str> {
    match tag("//")(s) {
        Ok((i,_)) => {
            match take_until("\n")(i) {
                Ok((_,o)) => {
                    let n: usize = o.len() + 2;
                    take(n)(s)
                }
                Err(e) => Err(e)
            }
        }
        Err(e) => Err(e)
    }
}

// ignore comments within strings, i.e. char* a = " // not a comment ";
fn skip_quoted_slice(s: &str) -> IResult<&str, &str> {
    match tag("\"")(s) {
        Ok((i,_)) => {
            match take_until("\"")(i) {
                Ok((_,o)) => {
                    let n: usize = o.len() + 2;
                    match take(n)(s) {
                        Ok((i,_)) => Ok((i,"")),
                        Err(e) => Err(e)
                    }
                }
                Err(e) => Err(e)
            }
        }
        Err(e) => Err(e)
    }
}

// advance slice by one element
fn advance_by_1(s: &str) -> IResult<&str, &str> {
    match take(1usize)(s) {
        Ok((i,_)) => Ok((i,"")),
        Err(e) => Err(e)
    }
}

fn parse(s: &str) -> IResult<&str,Vec<&str>> {
    fold_many0(
        alt((
            find_c_comment,  // try to enter C comment
            find_cpp_comment,  // try to enter C++ comment
            skip_quoted_slice,  // ignore comments within strings
            advance_by_1  // nothing found, keep looking
        )),
        Vec::new,
        | mut acc: Vec<_>, item | {
            if item.len() > 0  {
                acc.push(item);
            }
            else {}
            acc
        }
    )(s)
}

// parse the input string byte by byte
fn process_string(s: &str) {
    let(_,elements) = parse(&s).unwrap();
    for e in elements {
        println!("{}", e);
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
