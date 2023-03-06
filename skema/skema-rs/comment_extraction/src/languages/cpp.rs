use nom::{
    branch::alt, 
    bytes::complete::{tag, take, take_until}, 
    multi::fold_many0, 
    sequence::delimited, 
    IResult
};
use nom_locate::LocatedSpan;
use std::env;

type Span<'a> = LocatedSpan<&'a str>;

// find "/*", "*/" comments
fn locate_c_comment(input: Span) -> IResult<Span, Span> {
    delimited(tag("/*"), take_until("*/"), tag("*/"))(input)
}

// find "//", "\n" comments
fn locate_cpp_comment(input: Span) -> IResult<Span, Span> {
    delimited(tag("//"), take_until("\n"), tag("\n"))(input)
}

// removed quoted text
fn locate_quoted_slice(input: Span) -> IResult<Span, Span> {
    match delimited(tag("\""), take_until("\""), tag("\""))(input) {
        Ok((i, _)) => Ok((i, "".into())), 
        Err(e) => Err(e)
    }
}

// move to the next character
fn advance_by_1(input: Span) -> IResult<Span, Span> {
    match take(1usize)(input) {
        Ok((i, _)) => Ok((i, "".into())), 
        Err(e) => Err(e)
    }
}

// Return a vector of LocatedSpan objects with C and C++ style comments
fn parse(s: Span) -> IResult<Span, Vec<Span>> {
    fold_many0(
        alt((
            locate_c_comment, 
            locate_cpp_comment, 
            locate_quoted_slice, 
            advance_by_1
        )), 
        Vec::new, 
        | mut acc: Vec<Span>, span | {
            if span.fragment().len() > 0 {
                acc.push(span);
            }
            else {}
            acc
        }
    )(s)
}

// Parse the input string into a LocatedSpan vector and report the results
fn process_string(s: &str) {
    let span = LocatedSpan::new(s);
    let(_, elements) = parse(span).unwrap();
    for e in elements {
        println!("{:?}", e);
    }
}

// read a file as one big string and process it
fn process_file(file_path: &str) {
    println!("Processing file {:?}", file_path);
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

