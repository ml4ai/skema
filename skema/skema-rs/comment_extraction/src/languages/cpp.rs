use nom::{
    branch::alt, 
    bytes::complete::{tag, take, take_until}, 
    multi::fold_many0, 
    sequence::delimited, 
    IResult
};
use nom_locate::LocatedSpan;

/// Parse C/C++ code and output the comments along with their line numbers.  

type Span<'a> = LocatedSpan<&'a str>;

// find C and C++ style comments
fn parse_comment(input: Span) -> IResult<Span, Span> {
    alt((
        delimited(tag("/*"), take_until("*/"), tag("*/")), // C
        delimited(tag("//"), take_until("\n"), tag("\n"))  // C++ 
    ))(input)
}

// Move to next input element, bypassing quoted elements if found
fn parse_next(input: Span) -> IResult<Span, Span> {
    match alt((
        delimited(tag("\""), take_until("\""), tag("\"")), // quoted
        take(1usize)  // Move to next element 
    ))(input) {
        Ok((i, _)) => Ok((i, "".into())), 
        Err(e) => Err(e)
    }
}

// Return a vector of LocatedSpan objects with C and C++ style comments
fn parse(s: Span) -> IResult<Span, Vec<Span>> {
    fold_many0(
        alt((parse_comment, parse_next)), 
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

// Find the C and C++ style comments in the input string and report
// them along with their line numbers
fn process_string(s: &str) {
    let span = LocatedSpan::new(s);
    let(_, elements) = parse(span).unwrap();
    for e in elements {
        println!("{:?}", e);
    }
}

// read a file as one big string and process it
pub fn get_comments(file_path: &str) {
    let string = std::fs::read_to_string(file_path);
    match string {
        Ok(s) => process_string(&s), 
        Err(e) => println!("Error: {e:?}")
    }
}
