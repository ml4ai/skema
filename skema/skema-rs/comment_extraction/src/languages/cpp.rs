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
                    let n: usize = o.len() + 3;
                    take(n)(s)
                }
                Err(e) => Err(e)
            }
        }
        Err(e) => Err(e)
    }
}

fn parse(s: &str) -> IResult<&str,Vec<&str>> {
    fold_many0(
        alt((
            find_c_comment,  // try to enter C comment
            find_cpp_comment,  // try to enter C++ comment
            take(1usize)  // else advance slice by 1 element 
        )),
        Vec::new,
        | mut acc: Vec<_>, item | {
            acc.push(item);
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
